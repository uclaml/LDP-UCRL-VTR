import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import csv
import numpy as np

if __name__ == '__main__':

    LSVI = []
    LDP1 = []
    LDP2 = []
    LDP3 = []
    for i in range(10):
        with open(f'UCRL_{i}.csv', 'r', newline='') as csvfile:
            reader  = csv.reader(csvfile)
            for line in reader:
                temp = list(map(float, line))
                LSVI.append(temp)

        with open(f'UCRL10_{i}.csv', 'r', newline='') as csvfile:
            reader  = csv.reader(csvfile)
            for line in reader:
                temp = list(map(float, line))
                LDP1.append(temp)

        with open(f'UCRL1_{i}.csv', 'r', newline='') as csvfile:
            reader  = csv.reader(csvfile)
            for line in reader:
                temp = list(map(float, line))
                LDP2.append(temp)
        
        with open(f'UCRL01_{i}.csv', 'r', newline='') as csvfile:
            reader  = csv.reader(csvfile)
            for line in reader:
                temp = list(map(float, line))
                LDP3.append(temp)

    K = 10000
    meanLSVI = np.mean(LSVI, axis = 0)[:K]
    stdLSVI = np.std(LSVI, axis = 0)[:K]

    meanLDP1 = np.mean(LDP1, axis = 0)[:K]
    stdLSVI1 = np.std(LDP1, axis = 0)[:K]

    meanLDP2 = np.mean(LDP2, axis = 0)[:K]
    stdLSVI2 = np.std(LDP2, axis = 0)[:K]

    meanLDP3 = np.mean(LDP3, axis = 0)[:K]
    stdLSVI3 = np.std(LDP3, axis = 0)[:K]

    plt.plot(range(K), meanLSVI, color='purple', lw=0.5, ls='-', marker='o', ms = 0.1, label = r'UCRL-VTR')
    plt.fill_between(range(K), meanLSVI - stdLSVI, meanLSVI + stdLSVI, color=(229/256, 204/256, 249/256), alpha=0.3)

    # plt.plot(range(K), meanLDP1, color='green', lw=0.5, marker='^', ms = 0.1, label = r'LDP-UCRL-VTR, $\varepsilon = 10$')
    # plt.fill_between(range(K), meanLDP1 - stdLSVI1, meanLDP1 + stdLSVI1, color=(204/256, 236/256, 223/256), alpha=0.3)

    plt.plot(range(K), meanLDP2, color='blue', lw=0.5, ls='-', marker='o', ms = 0.1, label = r'LDP-UCRL-VTR, $\varepsilon = 1$')
    plt.fill_between(range(K), meanLDP2 - stdLSVI2, meanLDP2 + stdLSVI2, color=(191/256, 191/256, 255/256), alpha=0.3)

    plt.plot(range(K), meanLDP3, color='red', lw=0.5, ls='-', marker='o', ms = 0.2, label = r'LDP-UCRL-VTR, $\varepsilon = 0.1$')
    plt.fill_between(range(K), meanLDP3 - stdLSVI3, meanLDP3 + stdLSVI3, color=(255/256, 191/256, 191/256), alpha=0.3)
    


    
    plt.legend(loc = 2)

    # plt.legend()
    labelFont = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 10}
    plt.gca().ticklabel_format(style='scientific',scilimits=(0,0),useMathText=True)
    plt.xlabel(r'Episode($K$)', labelFont)
    plt.ylabel(r'Cumulative Regret($K$)', labelFont)
    plt.savefig(f'test.pdf')

