"""Example of THz field using field profile from Fig. 4 in https://doi.org/10.1364/OE.24.021059
(digitized figure using https://apps.automeris.io/wpd/)
"""

import numpy as np
import csv
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

with open("../data/thz_field.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=";")
    field = np.array(
        [[float(elem.replace(",", ".")) for elem in row] for row in csv_reader]
    )
    field -= [np.min(field.T[0]), 0]
    field /= [1, np.max(field.T[1])]
    THZ_FIELD_PROFILE = interp1d(*field.T, fill_value=0, bounds_error=False)


def thz_field(time_ps, peak_field=1):
    return THZ_FIELD_PROFILE(time_ps) * peak_field


if __name__ == "__main__":

    # Use example
    t = np.linspace(0, 20, 1000)
    plt.plot(t, thz_field(t, peak_field=30e6))
    plt.xlabel("time (ps)")
    plt.ylabel("THz field (V/m)")
    plt.show()
