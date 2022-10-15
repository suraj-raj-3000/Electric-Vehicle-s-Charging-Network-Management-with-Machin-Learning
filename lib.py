import matplotlib.pyplot as plt

# Now let us make a function to plot time series plots..
def plot_time_series(df, start=0, end=None, font_size=14, title_font_size=16, label=None, color="b"):

    plt.plot(df[start:end], label=label, c=color)
    plt.title("Time Series Plot", fontsize=title_font_size)
    if label:
        plt.legend(fontsize=font_size)
        plt.xlabel("Time", fontsize=font_size)
        plt.ylabel("Energy demand")
        plt.grid()
        plt.show()


def user_behaviour(caltech_ts):
    plt.figure(figsize=(10,5))
    plt.plot(caltech_ts["energyDemand"][130:400], label="Caltech Energy Demand")
    plt.title("Time Series Energy Demand")
    plt.xlabel("Time")
    plt.ylabel("Energy Demand")
    plt.legend()

def jpl_energy_demand(jpl_ts):
    plt.figure(figsize=(10,5))
    plt.plot(jpl_ts["energyDemand"][30:300], label="JPL Energy Demand")
    plt.title("Time Series Energy Demand")
    plt.xlabel("Time")
    plt.ylabel("Energy Demand")
    plt.legend()

def caltech_evse(caltech_ts):
    plt.figure(figsize=(10,5))
    plt.plot(caltech_ts["sessions"][130:400], label="Caltech sessions served per day")
    plt.title("Time Series sessions served at Caltech EVSE")
    plt.xlabel("Time")
    plt.ylabel("Sessions")
    plt.legend()

def jpl_evse(jpl_ts):
    plt.figure(figsize=(10,5))
    plt.plot(jpl_ts["sessions"][30:300], label="JPL sessions served per day")
    plt.title("Time Series sessions served at JPL EVSE")
    plt.xlabel("Time")
    plt.ylabel("Sessions")
    plt.legend()


def fre_paid_charging(a1,a2,a3,a4):
    fig, axes = plt.subplots(2,2, figsize=(10,10))
    fig.subplots_adjust(hspace=0.4, top=0.85)
    fig.suptitle("Arrival Time Analysis for Paid and Free Users on weekDays and weekEnds", fontsize=16)
    a1.hist(bins=20, ax=axes[0][0], label="weekDay")
    a2.hist(bins=20, ax=axes[0][1], label="weekDay")
    axes[0][0].set_title("Free Charging")
    axes[0][1].set_title("Paid Charging")
    axes[0][0].legend()
    axes[0][1].legend()
    a3.hist(bins=20, ax=axes[1][0], label="weekEnd")
    a4.hist(bins=20, ax=axes[1][1], label="weekEnd")
    axes[1][0].set_title("Free Charging")
    axes[1][1].set_title("Paid Charging")
    axes[1][0].legend()
    axes[1][1].legend()

def kWhDelivered_plot(simple_df):
    # Let us plot a scatter plot to furthur understand the dataset..
    plt.figure(figsize=(10,7))
    plt.scatter(x=simple_df["session_length"],y=simple_df["kWhDelivered"])
    plt.show()


def scatter_plot_energy(simple_df):
    plt.figure(figsize=(10,7))
    plt.scatter(x=simple_df["session_length"],y=simple_df["kWhDelivered"],alpha=0.1)
    plt.title("Scatter plot Energy consumed vs Session Length")
    plt.xlabel("Session Length")
    plt.ylabel("Energy Consumed")
    plt.show()

