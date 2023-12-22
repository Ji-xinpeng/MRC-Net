import matplotlib.pyplot as plt

class Record:
    def __init__(self):
        self.train_acc_top1 = []
        self.train_acc_top5 = []
        self.train_loss = []

        self.val_acc_top1 = []
        self.val_acc_top5 = []
        self.val_loss = []

    def get_size(self):
        print("train_acc_top1 size = ", len(self.train_acc_top1))
        print("train_acc_top5 size = ", len(self.train_acc_top5))
        print("train_loss size = ", len(self.train_loss))
        print("val_acc_top1 size = ", len(self.val_acc_top1))
        print("val_acc_top5 size = ", len(self.val_acc_top5))
        print("val_loss size = ", len(self.val_loss))


def plot(record):
    # 绘制准确率曲线 top1
    plt.plot(range(1, len(record.train_acc_top1) + 1), record.train_acc_top1, label='Train Acc top1')
    plt.plot(range(1, len(record.train_acc_top1) + 1), record.val_acc_top1, label='Test Acc top1')

    plt.xlabel('Epoch')
    plt.ylabel('Accuary')
    plt.title('Train and Test Accuary Top 1')
    plt.legend()
    plt.show()


    # 绘制损失曲线
    plt.plot(range(1, len(record.train_loss) + 1), record.train_loss, label='Train loss')
    plt.plot(range(1, len(record.val_loss) + 1), record.val_loss, label='Test loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.show()


    # 绘制准确率曲线 top1
    plt.plot(range(1, len(record.train_acc_top5) + 1), record.train_acc_top5, label='Train Acc top5')
    plt.plot(range(1, len(record.train_acc_top5) + 1), record.val_acc_top5, label='Test Acc top5')

    plt.xlabel('Epoch')
    plt.ylabel('Accuary')
    plt.title('Train and Test Accuary Top 5')
    plt.legend()
    plt.show()