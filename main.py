from model import RegressorModel

def main():
    model = RegressorModel(3, 64, 16)
    print(model)
    X = [
        [1, 2, 3],
    ]
    print(model.forward(X))


if __name__ == '__main__':
    main()