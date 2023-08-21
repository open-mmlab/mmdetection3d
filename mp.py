import multiprocessing
import time


class A(object):

    def __init__(self):
        self.a = []
        self.b = None

    def get_num_a(self, num):
        time.sleep(3)
        self.a = [0]
        self.get_num_b(num)
        print(self.a)
        return self.a

    def get_num_b(self, num):
        self.a.append(num)

    def sum(self):
        print(self.a)

    def run(self):
        # p1 = multiprocessing.Process(target=self.get_num_a)
        # p2 = multiprocessing.Process(target=self.get_num_b)
        # p1.start()
        # p2.start()
        # p1.join()
        # p2.join()
        p1 = multiprocessing.Pool(2)
        list_a = p1.map(self.get_num_a, [1, 2])
        self.sum()
        print(list_a)


if __name__ == '__main__':

    t1 = time.time()
    a = A()
    a.run()
    t2 = time.time()
    print('cost time :{}'.format(t2 - t1))
