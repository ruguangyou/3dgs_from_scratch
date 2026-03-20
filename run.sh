if [ "$1" = "train" ]; then
    python3 -m scripts.train
elif [ "$1" = "evaluate" ]; then
    python3 -m scripts.evaluate
elif [ "$1" = "test" ]; then
    python3 -m scripts.test
else
    echo "Usage: $0 {train|evaluate|test}"
    exit 1
fi