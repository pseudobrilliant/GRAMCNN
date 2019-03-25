import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pkl", "-p", help="tokenized data", required=True)
    args = parser.parse_args()




if __name__ == "__main__":
    main()