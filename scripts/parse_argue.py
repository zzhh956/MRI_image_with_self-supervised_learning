import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--train",
    help = "train",
    action = "store_true",
)

parser.add_argument(
    "--valid",
    help = "valid",
    action = "store_true",
)

parser.add_argument(
    "--test",
    help = "test",
    action = "store_true",
)

parser.add_argument(
    "--epoch",
    dest = "epoch_idx",
)