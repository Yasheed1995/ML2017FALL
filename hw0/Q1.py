import sys
__author__ = 'b04901025'


def main():
    filename = sys.argv[1]
    for line in open(filename, "r"):
        a = line.split()
        b = dict()
        c = set(a)
        for item in c:
            b[item] = a.count(item)
        d = sorted(b.items(), key=lambda x: x[1], reverse=True)

        count = 0
        fout = open("Q1.txt", "w")
        for i in d:
            fout.write(i[0] + ' ' + str(count) + ' ' + str(i[1]) + '\n')
            count += 1

if __name__ == '__main__':
    main()
