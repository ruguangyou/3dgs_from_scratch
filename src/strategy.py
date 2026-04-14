import logging
import torch
from src.gaussian import quaternion_to_rotation_matrix


class Strategy:
    refine_start_step = 500
    refine_stop_step = 15000
    refine_every_n_steps = 100
    reset_every_n_steps = 3000
    prune_opacity = 0.005  # prune gaussians with opacity below this threshold
    prune_scale = 0.1  # prune gaussians with scale above this threshold
    grow_grad = 0.0002  # grow new gaussians with gradient above this threshold
    split_scale = 0.01  # split gaussians with scale above this threshold, otherwise clone them
    max_gaussians = 1000000  # hard cap on total gaussian count to prevent OOM

    def __init__(self):
        self.grads = None
        self.count = None

    def adjust(self, learnable_params, optimizers, points_image, radii, width, height, step):
        if step > self.refine_stop_step:
            return

        self._update_state(points_image, radii, width, height)

        if step > self.refine_start_step and step % self.refine_every_n_steps == 0:
            n_prune = self._prune_gaussians(learnable_params, optimizers, step)
            N = learnable_params["means"].shape[0]
            logging.info(f"Step {step}: pruned {n_prune} gaussians, remaining {N} gaussians")
            # free GPU memory from pruned tensors immediately after pruning
            torch.cuda.empty_cache()

            n_clone, n_split = self._grow_gaussians(learnable_params, optimizers)
            N = learnable_params["means"].shape[0]
            logging.info(
                f"Step {step}: cloned {n_clone} gaussians, split {n_split} gaussians, total {N} gaussians"
            )

            # reset the state for the next round of refinement
            self.grads.zero_()
            self.count.zero_()

        if step > 0 and step % self.reset_every_n_steps == 0:
            self.reset_opacities(learnable_params, optimizers)

    def _update_state(self, points_image, radii, width, height):
        # points_image: (N, 2), radii: (N, 2)
        N = points_image.shape[0]
        grads_image = points_image.grad.clone()  # (N, 2)

        # scale gradients to NDC (normalized device coordinates) space
        #  u = width / 2 * (x_ndc + 1)
        #  v = height / 2 * (y_ndc + 1)
        #  dL/dx_ndc = dL/du * du/dx_ndc = dL/du * width / 2
        #  dL/dy_ndc = dL/dv * dv/dy_ndc = dL/dv * height / 2
        grads_image[:, 0] *= width / 2.0
        grads_image[:, 1] *= height / 2.0

        if self.grads is None:
            self.grads = torch.zeros(N, device=grads_image.device)
        if self.count is None:
            self.count = torch.zeros(N, device=grads_image.device)

        mask = (radii > 0.0).all(dim=1)  # (N,)
        selection = torch.where(mask)[0]  # (M,)
        self.grads.index_add_(0, selection, grads_image[selection].norm(dim=1))
        self.count.index_add_(0, selection, torch.ones_like(selection, dtype=torch.float))

    @torch.no_grad()
    def _prune_gaussians(self, learnable_params, optimizers, step):
        opacities = torch.sigmoid(learnable_params["opacities"])  # (N,)
        is_prune = opacities < self.prune_opacity
        if step > self.reset_every_n_steps:
            scales = torch.exp(learnable_params["scales"])  # (N, 3)
            is_prune = is_prune | (scales.max(dim=1).values > self.prune_scale)

        selection = torch.where(~is_prune)[0]
        for name in learnable_params.keys():
            param = learnable_params[name]
            new_param = torch.nn.Parameter(
                param[selection].clone(), requires_grad=param.requires_grad
            )
            learnable_params[name] = new_param

            optimizer = optimizers[name]
            for i in range(len(optimizer.param_groups)):
                # replace the old parameter with the new parameter in optimizer
                optimizer.param_groups[i]["params"] = [new_param]

                # update the optimizer's state for the new parameter
                if not hasattr(optimizer, "state"):
                    continue
                state = optimizer.state[param]
                del optimizer.state[param]
                for key in state.keys():
                    if key != "step":
                        state[key] = state[key][selection].clone()
                optimizer.state[new_param] = state

            # explicitly delete old param to release the GPU tensor immediately
            del param

        self.grads = self.grads[selection].clone()
        self.count = self.count[selection].clone()

        return is_prune.sum().item()

    @torch.no_grad()
    def _grow_gaussians(self, learnable_params, optimizers):
        grads = self.grads / self.count.clamp(min=1.0)  # (N,)
        scales = torch.exp(learnable_params["scales"])  # (N, 3)
        device = grads.device
        N = learnable_params["means"].shape[0]

        # enforce hard cap: if already at max, skip growing entirely
        if N >= self.max_gaussians:
            logging.warning(
                f"Gaussian count {N} reached max_gaussians={self.max_gaussians}, skipping growth."
            )
            return 0, 0

        is_grad_high = grads > self.grow_grad
        is_gaussian_small = scales.max(dim=1).values < self.split_scale
        is_clone = is_grad_high & is_gaussian_small
        is_split = is_grad_high & ~is_gaussian_small
        n_clone = is_clone.sum().item()
        n_split = is_split.sum().item()

        # cap cloning to not exceed max_gaussians
        budget = self.max_gaussians - N
        if n_clone > budget:
            selection_all = torch.where(is_clone)[0]
            perm = torch.randperm(n_clone, device=device)[:budget]
            is_clone = torch.zeros_like(is_clone)
            is_clone[selection_all[perm]] = True
            n_clone = budget

        if n_clone > 0:
            selection = torch.where(is_clone)[0]
            for name in learnable_params.keys():
                param = learnable_params[name]
                new_param = torch.nn.Parameter(
                    torch.cat([param, param[selection]], dim=0), requires_grad=param.requires_grad
                )
                learnable_params[name] = new_param

                optimizer = optimizers[name]
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]["params"] = [new_param]

                    if not hasattr(optimizer, "state"):
                        continue
                    state = optimizer.state[param]
                    del optimizer.state[param]
                    for key in state.keys():
                        if key != "step":
                            shape = state[key].shape
                            state[key] = torch.cat(
                                [state[key], torch.zeros((n_clone, *shape[1:]), device=device)],
                                dim=0,
                            )
                    optimizer.state[new_param] = state

                # explicitly delete old param to release the GPU tensor immediately
                del param

            self.grads = torch.cat([self.grads, self.grads[selection]], dim=0)
            self.count = torch.cat([self.count, self.count[selection]], dim=0)

        if n_split > 0:
            # take into account the new gaussians from cloning
            is_split = torch.cat(
                [is_split, torch.zeros(n_clone, device=device, dtype=torch.bool)], dim=0
            )
            selection = torch.where(is_split)[0]
            rest = torch.where(~is_split)[0]

            quaternions_split = torch.nn.functional.normalize(
                learnable_params["quaternions"][selection], dim=1
            )
            rotations_split = quaternion_to_rotation_matrix(quaternions_split)  # (n_split, 3, 3)
            scales_split = scales[selection]  # (n_split, 3)
            probe = torch.randn((2, n_split, 3), device=device)  # (2, n_split, 3)
            samples = (
                rotations_split.unsqueeze(0) @ (scales_split.unsqueeze(0) * probe).unsqueeze(-1)
            ).squeeze(-1)  # (2, n_split, 3)

            for name in learnable_params.keys():
                param = learnable_params[name]
                if name == "means":
                    param_split = (param[selection].unsqueeze(0) + samples).reshape(2 * n_split, 3)
                elif name == "scales":
                    param_split = torch.log(scales_split / 1.6).repeat(2, 1)
                else:
                    param_split = param[selection].repeat([2] + [1] * (param.dim() - 1))
                new_param = torch.nn.Parameter(
                    torch.cat([param[rest], param_split], dim=0), requires_grad=param.requires_grad
                )
                learnable_params[name] = new_param

                optimizer = optimizers[name]
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]["params"] = [new_param]

                    if not hasattr(optimizer, "state"):
                        continue
                    state = optimizer.state[param]
                    del optimizer.state[param]
                    for key in state.keys():
                        if key != "step":
                            shape = state[key].shape
                            state[key] = torch.cat(
                                [
                                    state[key][rest],
                                    torch.zeros((n_split * 2, *shape[1:]), device=device),
                                ],
                                dim=0,
                            )
                    optimizer.state[new_param] = state

                # explicitly delete old param to release the GPU tensor immediately
                del param

            self.grads = torch.cat([self.grads[rest], self.grads[selection].repeat(2)], dim=0)
            self.count = torch.cat([self.count[rest], self.count[selection].repeat(2)], dim=0)

        return n_clone, n_split

    @torch.no_grad()
    def reset_opacities(self, learnable_params, optimizers):
        opacities = learnable_params["opacities"]
        threshold = torch.logit(torch.tensor(self.prune_opacity * 2.0, device=opacities.device))
        new_opacities = torch.nn.Parameter(
            torch.clamp(opacities, max=threshold), requires_grad=opacities.requires_grad
        )
        learnable_params["opacities"] = new_opacities

        optimizer = optimizers["opacities"]
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]["params"] = [new_opacities]

            if not hasattr(optimizer, "state"):
                continue
            state = optimizer.state[opacities]
            del optimizer.state[opacities]
            for key in state.keys():
                if key != "step":
                    state[key] = torch.zeros_like(state[key])
            optimizer.state[new_opacities] = state
