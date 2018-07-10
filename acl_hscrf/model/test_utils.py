import utils

class Test:
    def __init__(self):
        pass

    def test_iob2(self):
        tags = ['O', 'O', 'I-LOC', 'I-LOC', 'O', 'O', 'I-PER', 'I-PER', 'I-PER', 'O']
        utils.iob2(tags)
        print(tags)

    def test_iob_iobes(self):
        tags = ['O', 'O', 'I-LOC', 'I-LOC', 'O', 'O', 'I-PER', 'I-PER', 'I-PER', 'O']
        result = utils.iob_iobes(tags)
        print(result)

    def main(self):
        self.test_iob2()
        self.test_iob_iobes()

if __name__ =="__main__":
    test = Test()
    test.main()