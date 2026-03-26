if [ "$1" = "train" ]; then
    python3 -m scripts.train
elif [ "$1" = "evaluate" ]; then
    python3 -m scripts.evaluate
elif [ "$1" = "test" ]; then
    python3 -m scripts.test
elif [ "$1" = "gradcheck" ]; then
    ./scripts/check_cuda_gradients.sh "${2:-3}"
else
    echo "Usage: $0 {train|evaluate|test|gradcheck [seeds]}"
    exit 1
fi