import matplotlib.pyplot as plt
import seaborn as sns
from components.epsilon_schedules import PolynomialDecaySchedule, LinearDecaySchedule, ExponentialDecaySchedule




if __name__ == '__main__':

    start = 0.2
    finish = 0.05
    steps = 4000000
    
    schedules = [
        PolynomialDecaySchedule(start, finish, steps, 0.3),
        PolynomialDecaySchedule(start, finish, steps, 0.5),
        PolynomialDecaySchedule(start, finish, steps, 0.7),
        PolynomialDecaySchedule(start, finish, steps, 1.5),
        PolynomialDecaySchedule(start, finish, steps, 2),
        LinearDecaySchedule(start, finish, steps),
        ExponentialDecaySchedule(start, finish, steps),
    ]

    values = [[] for _ in range(len(schedules))]

    for step in range(0, steps):
        for i, schedule in enumerate(schedules):
            value = schedule.eval(step)
            values[i].append(value)


    sns.set(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    plt.plot(values[0], label='Polynomial 0.3')
    plt.plot(values[1], label='Polynomial 0.5')
    plt.plot(values[2], label='Polynomial 0.7')
    plt.plot(values[3], label='Polynomial 1.5')
    plt.plot(values[4], label='Polynomial 2')
    plt.plot(values[5], label='Linear')
    plt.plot(values[6], label='Exponential')

    plt.title('Sigma decay schedule comparison')
    plt.xlabel('Step')
    plt.ylabel('Sigma')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./schedules.png', dpi=300)
    plt.close()
