
import pickle
import sys

def inspect(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Keys in {filename}:")
    for k in data.keys():
        print(f"  - {k}")

if __name__ == "__main__":
    inspect(sys.argv[1])
