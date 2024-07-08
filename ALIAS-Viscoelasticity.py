# import pandas
import numpy as np
from datetime import datetime
import glob
import os
import os.path
import shutil
import csv
import matplotlib.pyplot as plt
import matplotlib.legend_handler
import matplotlib.collections
import pandas as pd
# import matplotlib.cm as cm
from tkinter import *
from tkinter import scrolledtext
from tkinter import filedialog
# from tkinter.filedialog import asksaveasfilename
# import math
# import warnings
# from matplotlib.colors import colorConverter

# import scipy.interpolate
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
# from sklearn.linear_model import LinearRegression
from sklearn import linear_model
# ----------------------------------------------------------------------------------------------------------------------#
#BASIC FUNCTIONS
def Spacer():
    """Creates a spacer (---) in GUI dialogue window."""
    txt.insert(END, "-"*134)
    txt.insert(END, "\n")
    txt.update()
    txt.see("end")

def Create_Output_Folder(new_folder_name):
    """Creates a new directory with given input folder name."""
    try:
        os.mkdir(new_folder_name) # Make new directory

    # Error handling to avoid stopping code
    except OSError:
        txt.insert(END, "Error: folder {} already present.\n".format(new_folder_name))
        txt.update()
        txt.see("end")

    else:
        txt.insert(END, "Directory {} successfully created.\n".format(new_folder_name))
        txt.update()
        txt.see("end")


def Find_Experiments(path_filename):
    """Divides txt file containing multiple (array or single) measurements into individual txt files for each measurement."""
    # Save txt files in new folder with the name of the txt file
    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)
    folder_name = filename_noext
    folder_path = "{}/{}".format(path, folder_name)
    Create_Output_Folder(folder_path) #This is for the experiment


    inputfile = open(path_filename, 'r') #, encoding='gbk'
    txt.insert(END, "Working directory:\n{}\n\n".format(path))
    txt.insert(END, "File opened: {}\n\n".format(filename))
    txt.update()
    txt.see("end")
    Spacer()

    fileno = -1
    outfile = open(f"{folder_path}/{filename_noext}_{fileno}.txt", "w") #This is the new f-string method of Python 3.6
    txt.insert(END, f"File successfully created: {filename_noext}_{fileno}.txt\n\n")
    txt.update()
    txt.see("end")
    for line in inputfile:
        if not line.strip():
            fileno += 1
            outfile.close()
            outfile = open(f"{folder_path}/{filename_noext}_{fileno}.txt", "w")
            txt.insert(END, f"File successfully created: {filename_noext}_{fileno}.txt\n\n")
            txt.update()
            txt.see("end")
        else:
            outfile.write(line)
    outfile.close()
    inputfile.close()

    #Delete data_-1.txt file which is useless
    os.remove(f"{folder_path}/{filename_noext}_-1.txt")
    txt.insert(END, f"File {folder_path}/{filename_noext}_-1.txt deleted.\n")
    txt.update()
    txt.see("end")


    txt.insert(END, "Individual measurements successfully detected.\n")
    txt.insert(END, "Detected {} measurements.\n\n".format(fileno+1))
    txt.update()
    txt.see("end")
    return path, filename, filename_noext, extension, folder_path

def Divide_Array_Data(path_filename, folder_path_Array_Measurements):
    """Divides a txt file containing an array of indentation measurements into individual csv files for each indentation."""
    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)
    # folder_path_Array_Measurements = path + f"/{filename_noext}_Measurements"
    # Create_Output_Folder(folder_path_Array_Measurements)

    #Convert data to df
    df = pd.read_table(path_filename, low_memory=False, encoding = "unicode_escape", on_bad_lines='skip', delim_whitespace=True, names=("Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]", "Pos Z [um]", "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]", "Gripper [V]", "Voltage A [V]", "Voltage B [V]", "Temperature [oC]", "Sample Displacement [um]"))
    txt.insert(END, f"DF: {df}\n")
    df = df[~df["Index [#]"].isin(['//'])]  # Drop the rows that contain the comments to keep only the numbers    , encoding = "iso-8859-1"
    #df = df.replace('//', np.nan, regex=True).replace('=', np.nan, regex=True)
    df = df.dropna(how='all')  # to drop if all values in the row are nan
    df = df.astype(float)  # Change data from object to float

    #txt.insert(END, f"DF: {df}\n")

    num_mes = int(df["Index [#]"].max() + 1)
    txt.insert(END, f"Number of individual measurements found: {str(num_mes)}\n")
    txt.update()
    txt.see("end")

    grouped = df.groupby(["Index [#]"])  # Group full database by measurement#

    for num in range(num_mes):
        #txt.insert(END, f"{num + 1}) Plotting file {filename_noext}_{str(num + 1)}...\n")
        #txt.update()
        #txt.see("end")

        group = pd.DataFrame(grouped.get_group(num))  # Get a single experiment

        group.to_csv(f"{folder_path_Array_Measurements}/{filename_noext}_{str(num + 1)}.csv", index=False)  # Save all measurements as *.csv, index=False avoids saving index

def DetermineIndentationDepth_DMA(path_filename, ratio_BL_points, threshold_constant): # DMA
    """Determines the indentation depth used for a DMA measurement from txt file of measurement recording."""
    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)
    txt.insert(END, "Analysing indentation depth in file {}...\n".format(filename_noext))
    Spacer()
    txt.update()
    txt.see("end")
    folder_path_Indentation_Depth = path + f"/{filename_noext}_Indentation_Depth_Analysis"
    Create_Output_Folder(folder_path_Indentation_Depth)

    df = pd.read_table(path_filename, encoding = "iso-8859-1", on_bad_lines='skip', low_memory=False, delim_whitespace=True, names=("Time [s]", "Pos X [um]", "Pos Y [um]", "Pos Z [um]", "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]", "Gripper [V]", "Voltage A [V]", "Voltage B [V]", "Temperature [oC]"))
    # Clean up txt file to save a proper csv file
    df = df[~df["Time [s]"].isin(['//'])]  # Drop the rows that contain the comments to keep only the numbers
    df = df.dropna(how='all')  # to drop if all values in the row are nan
    df = df.astype(float)  # Change data from object to float

    posZ = df["Pos Z [um]"].to_list()
    Time = df["Time [s]"].to_list()
    Force = df["Force A [uN]"].to_list()
    piezoZ = df["Piezo Z [um]"].to_list()

    TimeZero = [x - Time[0] for x in Time] #Normalises the time to start at 0
    TrimIndex = sum(map(lambda x : x<100, TimeZero))

    TrimmedTime = TimeZero[0 : TrimIndex] #Trim parameters to first 100 seconds of measurement (sometimes retraction noise also appears to show indentation)
    TrimmedPosZ = posZ[0 : TrimIndex]
    TrimmedForce = Force[0 : TrimIndex]

    deltaposZ = [posZ[n]-posZ[n-1] for n in range(1,len(TrimmedPosZ))] # Calculates difference between each adjacent value of posZ
    MovingIndex = [index for index, value in enumerate(deltaposZ) if value < -0.003] # Determine which portion of the data is active indentation (0.003 is an empirically chosen value)

    SelPosZ = [TrimmedPosZ[i] for i in MovingIndex] #Select only data where probe is moving towards sample
    SelTime = [TrimmedTime[i] for i in MovingIndex]
    SelForce = [TrimmedForce[i] for i in MovingIndex]


    df.to_csv(f"{folder_path_Indentation_Depth}/{filename_noext}_RawData.csv", index=False)  # Save all measurements as *.csv, index=False avoids saving index


    Dict = {'Time (s)':SelTime, 'Force (uN)':SelForce} #Define new df for CP analysis
    DMA_CP_df = pd.DataFrame(Dict)

    # BASELINE
    numpoint_baseline, _ = DMA_CP_df.shape
    numpoint_baseline = round(numpoint_baseline*ratio_BL_points) #Determine number of points for baseline is 1/8 of datapoints of df
    baseline = pd.DataFrame(DMA_CP_df.iloc[0:numpoint_baseline])
    x_bl = pd.DataFrame(baseline["Time (s)"])
    y_bl = pd.DataFrame(baseline["Force (uN)"])
    regr = linear_model.LinearRegression() #Check that x and y have a shape (n, 1)
    regr.fit(x_bl, y_bl)
    DMA_CP_df["Corr Force (uN)"] = DMA_CP_df["Force (uN)"]-(regr.intercept_[0]+(regr.coef_[0][0]*DMA_CP_df["Time (s)"])) #y=a+bx; add new column to df with corrected force values
    DMA_CP_Force_col = DMA_CP_df.columns.get_loc("Corr Force (uN)")


    DMA_CP_BL_mean = DMA_CP_df.iloc[0:int(DMA_CP_df.shape[0] * ratio_BL_points), DMA_CP_Force_col].mean() #Determine mean and st dev of baselined force data
    DMA_CP_BL_std = DMA_CP_df.iloc[0:int(DMA_CP_df.shape[0] * ratio_BL_points), DMA_CP_Force_col].std()
    txt.insert(END, f"Force baseline mean and std: {round(DMA_CP_BL_mean, 15)} +/- {round(DMA_CP_BL_std, 15)} uN\n")

    DMA_CP_Index = list(DMA_ForceValue > (DMA_CP_BL_mean+(DMA_CP_BL_std * threshold_constant)) or DMA_ForceValue < (DMA_CP_BL_mean-(DMA_CP_BL_std * threshold_constant)) for DMA_ForceValue in SelForce).index(True)
    txt.insert(END, f"Contact point index: {round(DMA_CP_Index, 15)}\n")

    DMA_IndentationDepth = -(min(SelPosZ)-SelPosZ[DMA_CP_Index])
    txt.insert(END, f"Indentation Depth: {round(DMA_IndentationDepth, 3)} um\n")


    #Data Plotting (Raw Data)
    x, y1, y2, y3 = (df["Time [s]"]-Time[0]), df["Pos Z [um]"], piezoZ, Force
    mask = (y1 <= (np.mean(posZ)+50))
    fig, (ax1, ax3) = plt.subplots(2, 1, sharex='all')   #sharex=True)

    ax1.set_xlabel('Time (s)', fontsize=18)
    ax1.set_ylabel('Piezo Z (um)', fontsize=18)
    lns1 = ax1.plot(x, y2, color='black', label='Piezo Z')
    ax1.tick_params(axis = 'y', labelcolor='black')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Force (uN)', fontsize=18, color='red')
    lns2 = ax2.plot(x, y3, color='red', label='Force')
    ax2.tick_params(axis = 'y', labelcolor='red')

    ax3.set_xlabel('Time (s)', fontsize=18)
    ax3.set_ylabel('Pos Z (um)', fontsize=18, color='blue')
    lns3 = ax3.plot(x[mask], y1[mask], color='blue', label='Pos Z')
    ax3.tick_params(axis = 'y', labelcolor='blue')

    plt.yticks(fontsize=12)
    plt.title("{}".format(f"{filename_noext}_Raw Data"), fontsize=20)
    fig.tight_layout()
    plt.savefig(f"{folder_path_Indentation_Depth}/{filename_noext}_Raw_Data.png", bbox_inches='tight', dpi=300)
    plt.close('all') #To close all figures and save memory - max # of figures before warning: 20

    #Data Plotting (Depth Analysis)
    max_moving_index_plotting = round(max(MovingIndex)*1.1) #Factor of 1.1 included to expand axes slightly beyond maximum indentation point
    max_moving_index = max(MovingIndex)

    x, y1, y2 = TrimmedTime[0:max_moving_index_plotting], TrimmedPosZ[0:max_moving_index_plotting], TrimmedForce[0:max_moving_index_plotting]
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)', fontsize=18)
    ax1.set_ylabel('Pos Z (um)', fontsize=18)
    lns1 = ax1.plot(x, y1, color='black', label='Pos Z')
    ax1.tick_params(axis = 'y', labelcolor='black')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Force (uN)', fontsize=18, color='red')
    lns2 = ax2.plot(x, y2, color='red', label='Force')
    lns3 = ax2.scatter(TrimmedTime[max_moving_index-1], TrimmedForce[max_moving_index-1], color='blue', s=20, label='Max indentation')
    lns4 = ax2.scatter(SelTime[DMA_CP_Index], SelForce[DMA_CP_Index], color='green', s=20, label='Contact point')
    ax2.tick_params(axis = 'y', labelcolor='red')

    ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.6))
    ax2.legend(loc='upper left', bbox_to_anchor=(0., 0.5))

    plt.yticks(fontsize=12)
    plt.title("{}".format(f"{filename_noext}_Indentation Depth Analysis"), fontsize=20)
    fig.tight_layout()
    plt.savefig(f"{folder_path_Indentation_Depth}/{filename_noext}_Indentation_Depth_Analysis.png", bbox_inches='tight', dpi=300)
    plt.close('all') #To close all figures and save memory - max # of figures before warning: 20

    txt.insert(END, f"Analysis of indentation depth complete.\n")
    Spacer()
    txt.update()
    txt.see("end")

def DMA_Analysis(path_filename, R, v_sample, Ind_Dep_um):
    """Analyses the results txt of a DMA measurement (using input values of probe radius, Poisson's ratio, and indentation depth) to determine the viscoelastic moduli as a function of frequency."""
    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)
    folder_path_DMA_Analysis = path + f"/{filename_noext}_DMA Analysis"
    Create_Output_Folder(folder_path_DMA_Analysis)

    df = pd.read_table(path_filename, encoding = "iso-8859-1", on_bad_lines='skip', low_memory=False, delim_whitespace=True, names=("Index [#]", "Time [s]", "Frequency [Hz]", "Mean force [uN]", "Mean position [um]", "Amplitude force [uN]", "Amplitude pos [um]", "Stiffness [N/m]", "Phase shift [degree]", "Real stiffness [N/m]", "Im. Stiffness [N/m]"))
    # Clean up txt file
    df = df[~df["Index [#]"].isin(['//'])]  # Drop the rows that contain the comments to keep only the numbers
    df = df.dropna(how='all')  # to drop if all values in the row are nan
    df = df.astype(float)  # Change data from object to float

    #df.to_csv(f"{folder_path_DMA_Analysis}/{filename_noext}_RawData.csv", index=False)  # Save all measurements as *.csv, index=False avoids saving index
    #df.to_csv(f"{path}/{filename_noext}_RawData.csv", index=False)  # Save all measurements as *.csv, index=False avoids saving index

    #Parameter definitions
    df["Amplitude F [N]"] = df["Amplitude force [uN]"] * 10 ** (-6)
    df["Amplitude Pos [m]"] = df["Amplitude pos [um]"] * 10 ** (-6)
    #df["Mean Pos [m]"] = df["Mean position [um]"] * 10 ** (-6)
    R_m = R * 10 ** (-6)
    Ind_Dep_m = (Ind_Dep_um + df["Mean position [um]"].mean()) * 10 ** (-6)
    Contact_R = (R_m * Ind_Dep_m) ** 0.5
    Contact_A = np.pi * (Contact_R ** 2)

    df["Area Normalised Frequency [m2 s-1]"] = df["Frequency [Hz]"] * (Contact_R ** 2)

    #Modulus calculations
    df["Storage Modulus [Pa]"] = round((1-(v_sample ** 2)) * (df["Amplitude F [N]"] / df["Amplitude Pos [m]"]) * (np.cos(np.radians(df["Phase shift [degree]"]))) / (2 * ((Ind_Dep_m * R_m) ** 0.5)), 2)
    df["Loss Modulus [Pa]"] = round((1-(v_sample ** 2)) * (df["Amplitude F [N]"] / df["Amplitude Pos [m]"]) * (np.sin(np.radians(df["Phase shift [degree]"]))) / (2 * ((Ind_Dep_m * R_m) ** 0.5)), 2)
    # df["Storage Modulus [Pa]"] = round((1-(v_sample ** 2)) * (df["Amplitude F [N]"] / df["Amplitude Pos [m]"]) * (np.cos(np.radians(df["Phase shift [degree]"]))) / (2 * (((df["Mean Pos [m]"] + Ind_Dep_m) * R_m) ** 0.5)), 6)
    # df["Loss Modulus [Pa]"] = round((1-(v_sample ** 2)) * (df["Amplitude F [N]"] / df["Amplitude Pos [m]"]) * (np.sin(np.radians(df["Phase shift [degree]"]))) / (2 * (((df["Mean Pos [m]"] + Ind_Dep_m) * R_m) ** 0.5)), 6)
    df["Loss Tangent"] = round(np.tan(np.radians(df["Phase shift [degree]"])), 5)

    df.to_csv(f"{folder_path_DMA_Analysis}/{filename_noext}_DMA_Analysis.csv", index=False)  # Save all data as *.csv, index=False avoids saving index

    #Data Plotting
    x, y1, y2, y3 = df["Frequency [Hz]"], (df["Storage Modulus [Pa]"]/1000), (df["Loss Modulus [Pa]"]/1000), df["Loss Tangent"]
    mask = (x <= 200) #Limits data to Frequency < 200 Hz as data is noisy above this value
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Frequency (Hz)', fontsize=18)
    ax1.set_ylabel('Modulus (kPa)', fontsize=18)
    lns1 = ax1.plot(x[mask], y1[mask], color='black', label='Storage Modulus')
    lns2 = ax1.plot(x[mask], y2[mask], color='blue', label='Loss Modulus')
    ax1.tick_params(axis = 'y', labelcolor='black')
    ax1.set_xscale('log')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss Tangent', fontsize=18, color='red')
    lns3 = ax2.plot(x[mask], y3[mask], color='red', label='Loss Tangent')
    ax2.tick_params(axis = 'y', labelcolor='red')

    lns = lns1+lns2+lns3 #To handle combined figure legend
    labs = [l.get_label() for l in lns] #To handle combined figure legend
    ax1.legend(lns, labs, loc=0, fontsize=12) #To handle combined figure legend
    plt.yticks(fontsize=12)
    plt.title("{}".format(f"{filename_noext}_DMA_Analysis"), fontsize=20)
    fig.tight_layout()
    plt.savefig(f"{folder_path_DMA_Analysis}/{filename_noext}_DMA_Analysis.png", bbox_inches='tight', dpi=300)
    plt.close('all') #To close all figures and save memory - max # of figures before warning: 20

    #Add experimental parameters to csv file
    file = open(f"{folder_path_DMA_Analysis}/{filename_noext}_DMA_Analysis.csv", "w")
    file.write("Probe radius:, {}, um\n".format(round((R_m*1e6), 3)))
    file.write("Mean indentation depth:, {}, um\n".format(round(Ind_Dep_m*1e6, 3)))
    file.write("Sample Poisson's Ratio:, {}\n".format(v_sample))
    file.write("Contact radius:, {}, m\n".format(round(Contact_R, 9)))
    file.write("Contact area:, {}, m2\n\n\n".format(round(Contact_A, 14)))
    df.to_csv(file, index=False, lineterminator='\n') #lineterminator='\n' is necessary otherwise it creates an empty line between each record.
    file.close()

def DetermineIndentationDepth_F_relax(df, path, filename, correct_drift):
    global Time_CP_Sel, Force_CP_Sel, fit_result, cp, a, b, exponent, c, d, e
    filename_noext, extension = os.path.splitext(filename)
    folder_path_Indentation_Depth = path + f"/Indentation_Depth_Analysis"

    Time_col = df.columns.get_loc("Time [s]")  # Get column number
    Force_col = df.columns.get_loc("Force A [uN]")  # Get column number
    PiezoZ_col = df.columns.get_loc("Piezo Z [um]")  # Get column number
    Phase_col = df.columns.get_loc("Phase [#]")

    Time = np.array(df.iloc[:, Time_col]) #define array of time data
    Force = np.array(df.iloc[:, Force_col])  #define array of force data
    PiezoZ = np.array(df.iloc[:, PiezoZ_col])  # define array of Piezo Z data
    Phase = np.array(df.iloc[:, Phase_col])

    # DETERMINE THE EQUATION OF A STRAIGHT LINE THROUGH FIRST AND LAST DATA POINTS OF MEASUREMENT TO CORRECT FOR FORCE DRIFT
    meanF_start = np.mean(Force[0:5])
    meanF_end = np.mean(Force[-6:-1])
    meanT_start = np.mean(Time[0:5])
    meanT_end = np.mean(Time[-6:-1])
    correction_gradient = (meanF_end-meanF_start)/(meanT_end-meanT_start)
    correction_intercept = meanF_start - (correction_gradient * meanT_start)

    Force_corrected = Force - (correction_gradient * Time + correction_intercept)

    if correct_drift == 1:
        Force = Force_corrected
    elif correct_drift == 0:
        pass

    # DETERMINE MEASUREMENT PHASES, AND CALCULATE DISPLACEMENT DURING MEASUREMENT
    F_max_index = np.argmax(Force) # index of maximum force as the change from loading to holding phases
    # F_max_index = np.array([np.where(Phase >= 3)]).min() # Alternatively define as where phase column changes from 2 - 3 (better for noisy data, i.e. low force values)

    # Define move back index (i.e. when probe retracts from sample) based on experimental phase
    MoveBack_Index = np.array([np.where(Phase >= 4)]).min(initial=np._NoValue)

    # Optional index adjustment for F_max_index:
    # Occasionally piezoscanner will overshoot then return slightly - the sharp decrease in F from this process should be excluded from analysis
    meanPiezoZ, sdPiezoZ = np.mean(PiezoZ[F_max_index:MoveBack_Index]), np.std(PiezoZ[F_max_index:MoveBack_Index]) # Calculate mean + SD of piezo Z channel during constant indentation
    PiezoZmax = meanPiezoZ + 100 * sdPiezoZ # Set a threshold piezo Z value as (mean + SD * multiple). 100 as default has no effect on data.
    Index_Adjustment = np.array([np.where(PiezoZ[F_max_index:MoveBack_Index] < PiezoZmax)]).min(initial=np._NoValue) # First index of PiezoZ < PiezoZmax (i.e. where PiezoZ is stable)
    PiezoZ_Max_index = F_max_index + Index_Adjustment # Adjust F_max_index to where PiezoZ is stable

    Displacement_Max_um = PiezoZ[PiezoZ_Max_index] # Define maximum displacement (where relaxation starts)
    #Displacement_Max_um = PiezoZ[F_max_index] #defines maximum displacement (where relaxation starts)
    txt.insert(END, "Maximum F index: {}\n\n".format(str(F_max_index)))
    txt.insert(END, "Maximum displacement: {} um\n\n".format(str(round(Displacement_Max_um, 3))))
    txt.update()
    txt.see("end")

    # CALCULATE CONTACT POINT:

    # DETERMINE SMOOTHING WINDOW BASED ON NUMBER OF DATA POINTS DURING LOADING
    SG_windowlength = round(len(Force[0:F_max_index])/15)
    if SG_windowlength % 2 == 0: # Ensures window length is odd (required for Savitsky-Golay filter)
        SG_windowlength = SG_windowlength + 1
    if SG_windowlength < 3: # Ensures window length is longer than polynomial order
        SG_windowlength = 3
    txt.insert(END, "SG window length: {}\n\n".format(str(SG_windowlength)))

    # SMOOTH DATA AND ESTIMATE CP AS MAXIMUM IN 2ND DERIVATIVE
    Force_Interp = savgol_filter((Force[0:F_max_index]), SG_windowlength, 2) # Interpolate data (Savitsky Golay)
    Est_SecondDeriv_Force = savgol_filter((Force[0:F_max_index]), SG_windowlength, 2, deriv=2) # Interpolate and calculate 2nd derivative of data
    Est_CP_index = np.argmax(Est_SecondDeriv_Force[0:F_max_index]) # Estimates index of contact point as maximum in 2nd derivative of smoothed data

    # SELECT REGION AROUND ESTIMATED CP FOR MORE ACCURATE FITTING
    # nb. estimated CP tends to be later than actual CP so window is skewed towards before estimated CP.
    CP_window = round((len(Force[0:F_max_index]))/8)
    CP_Selection_Min = Est_CP_index - 3*CP_window # If CP determination is inconsistent, selection min/max can be varied.
    CP_Selection_Max = Est_CP_index + CP_window
    if CP_Selection_Max > F_max_index: # Bound CP window between start of data and F_max_index
        CP_Selection_Max = F_max_index
    if CP_Selection_Min < 1:
        CP_Selection_Min = 1

    # PIECEWISE FITTING TO GIVE ACCURATE CONTACT POINT:
    # Selected data around the estimated CP are fit to linear and exponential regions with the
    # switchpoint between regions a fitting parameter which should be equivalent to the CP.
    try:
        Time_CP_Sel = Time[CP_Selection_Min:CP_Selection_Max]-Time[0]
        Force_CP_Sel = Force[CP_Selection_Min:CP_Selection_Max]-Force[0]

        from symfit import parameters, variables, Fit, Piecewise, Eq # exp,
        # from symfit.core.minimizers import DifferentialEvolution, NelderMead, BFGS, ScipyConstrainedMinimize # Alternative minimisers for symfit module

        # Define fitting parameters and x/y input (time/force)
        t, y = variables('t, y')
        a, b, c, d, e, cp = parameters('a, b, c, d, e, cp')

        # Help the fit by bounding the switchpoint between the models and using the estimated CP as starting switchpoint
        cp.min = np.amin(Time_CP_Sel)
        cp.max = np.amax(Time_CP_Sel)
        exponent = 1.5 # Vary exponent if needed (1.5 by default)
        cp.value = np.mean(Time_CP_Sel)

        # Make a piecewise model. Linear when t < cp, exponential when t >= cp.
        # Exponent can be varied but 1.5 is chosen based on the Hertz model (assuming time scales linearly with displacement)
        y1 = a * t + b
        y2 = c * t ** exponent + d * t + e
        model = {y: Piecewise((y1, t < cp), (y2, t >= cp))}

        # As a constraint, we demand equality between the two models at the point cp
        # to do this, we substitute t -> cp and demand equality using `Eq`
        constraints = [Eq(y1.subs({t: cp}), y2.subs({t: cp}))]

        # Alternative constraints: continuous derivative at point cp
        # constraints = [Eq(y1.diff(t).subs({t: cp}), y2.diff(t).subs({t: cp})), Eq(y1.subs({t: cp}), y2.subs({t: cp}))]

        # FIT DATA TO PIECEWISE MODEL WITH GIVEN CONSTRAINTS
        # The determined value of 'cp' is within 'fit_result.
        # Fitting methods:

        # 1. Using standard minimizer (least-squares) is fastest but may find local minima influenced by initial parameters:
        fit = Fit(model, t=Time_CP_Sel, y=Force_CP_Sel, constraints=constraints)

        # 2. BasinHopping algorithm is more computationally expensive but should do a better job of finding global minimum:
        # from symfit.core.minimizers import BasinHopping
        # fit = Fit(model, t=Time_CP_Sel, y=Force_CP_Sel, constraints=constraints, minimizer=BasinHopping)

        fit_result = fit.execute()

        # Determine the index (from the full df) of the CP, as the value before t > t(cp)
        Fitted_CP_index = (next(x for x, val in enumerate(Time-Time[0])
                             if val > (fit_result.value(cp)))) - 1

        fitted_cp = 1

    except(ValueError): #
        Fitted_CP_index = Est_CP_index
        fitted_cp = 0
        txt.insert(END, "Contact point fitting failed. Using estimated contact point from maximum second derivative.")
        Spacer()

    # TO CHECK CP FITTING
    # txt.insert(END, "CP fit index: {} [s]\n\n".format(str((Fitted_CP_index))))
    # txt.insert(END, "CP fit result: {} [s]\n\n".format(str((fit_result))))
    # txt.insert(END, "CP time fitted: {} [s]\n\n".format(str((fit_result.value(cp)))))

    # Determine parameters at CP
    fitted_cp_time = Time[Fitted_CP_index]
    F_Min_uN = Force[Fitted_CP_index]
    Displacement_Min_um = PiezoZ[Fitted_CP_index]
    Max_Indentation_Depth_um = float(Displacement_Max_um - Displacement_Min_um) # determines maximum indentation depth
    txt.insert(END, "Maximum indentation depth: {} um\n\n".format(str(np.round(Max_Indentation_Depth_um, 3))))
    txt.update()
    txt.see("end")

    Ramp_Time = float(Time[F_max_index]-Time[Fitted_CP_index])

    #Add calculated values to dataframe and save as csv
    df['Filtered time [uN}'] = pd.Series(Time[:F_max_index])
    df['Filtered force [uN]'] = pd.Series(Force[:F_max_index])
    df['SG Force [uN]'] = pd.Series(Force_Interp)
    df['Est Force 2nd derivative'] = pd.Series(Est_SecondDeriv_Force)
    # df['Force 2nd derivative'] = pd.Series(SecondDeriv_Force)
    df['Filtered displacement [um]'] = pd.Series(PiezoZ[:F_max_index])
    df.to_csv(f"{folder_path_Indentation_Depth}/{filename_noext}_IndentationDepthCalculation.csv", index=False)  # Save all measurements as *.csv, index=False avoids saving index

    # PLOT INDENTATION DEPTH DETERMINATION
    # 1. Estimated CP based on 2nd derivative
    x, y2, y3 = (Time[:F_max_index]-Time[0]), Est_SecondDeriv_Force, Force[:F_max_index]
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)', fontsize=18)
    ax1.set_ylabel('Force derivative', fontsize=18)
    #lns1 = ax1.plot(x, y1, color='black', label='Est 1st Derivative')
    lns2 = ax1.plot(x, y2, color='blue', label='Est 2nd Derivative')
    ax1.tick_params(axis = 'y', labelcolor='black')
    ax1.set_xscale('linear')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Force (uN)', fontsize=18, color='red')
    lns3 = ax2.plot(x, y3, color='red', label='Force_SG')
    lns4 = ax2.axvline(x=(Time[Est_CP_index]-Time[0]), color='k', linestyle='--', label='Contact Point')
    ax2.tick_params(axis = 'y', labelcolor='red')

    lns = lns2+lns3 #To handle combined figure legend
    labs = [l.get_label() for l in lns] #To handle combined figure legend
    ax1.legend(lns, labs, loc=0, fontsize=12) #To handle combined figure legend
    plt.yticks(fontsize=12)
    plt.title("{}".format(f"{filename_noext}_CP_approx"), fontsize=20)
    fig.tight_layout()
    plt.savefig(f"{folder_path_Indentation_Depth}/{filename_noext}_CP_approx.png", bbox_inches='tight', dpi=300)
    plt.close('all')

    # 2. Piecewise CP fitting (if successful)
    if fitted_cp == 1:
        xi, yi = Time_CP_Sel, Force_CP_Sel
        xii = np.linspace(np.amin(Time_CP_Sel), fit_result.value(cp), num=200)
        yii = xii * fit_result.value(a) + fit_result.value(b)
        xiii = np.linspace(fit_result.value(cp), np.amax(Time_CP_Sel), num=200)
        yiii = xiii ** exponent * fit_result.value(c) + fit_result.value(d) * xiii + fit_result.value(e) # Exponent to match fitting equation
        fig, ax1 = plt.subplots()

        txt.insert(END, "CP fit result params: {}\n\n".format(str((fit_result.params))))

        ax1.set_xlabel('Time (s)', fontsize=18)
        ax1.set_ylabel('Force (uN)', fontsize=18)
        lns1 = ax1.plot(xi, yi, 'ko', label='Raw data') #color='black'
        #lns2 = ax1.plot(xii, model(t=xii, **fit_result.params).y, color='blue', linestyle='--', label='Piecewise Fit')
        lns2 = ax1.plot(xii, yii, color='blue', linestyle='-', label='Piecewise Fit - Linear')
        lns3 = ax1.plot(xiii, yiii, color='red', linestyle='-', label='Piecewise Fit - Exp')
        ax1.tick_params(axis = 'y', labelcolor='black')
        ax1.set_xscale('linear')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        lns4 = ax1.axvline(x=fit_result.value(cp), color='g', linestyle='--', label='Contact Point (Fitted)')

        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Force (uN)', fontsize=18, color='red')
        # lns3 = ax2.plot(x1, y2, color='red', label='Force')
        # lns4 = ax2.plot(x2, y4, 'b--', label='Force_Interpolated')
        # lns5 = ax2.axvline(x=(CP_time-Time[0]), color='k', linestyle='--', label='Contact Point')
        # ax2.tick_params(axis = 'y', labelcolor='red')

        # lns = lns1+lns3 #To handle combined figure legend
        # labs = [l.get_label() for l in lns] #To handle combined figure legend
        # ax1.legend(lns, labs, loc=0, fontsize=12) #To handle combined figure legend
        plt.yticks(fontsize=12)
        plt.title("{}".format(f"{filename_noext}_CP_precise"), fontsize=20)
        fig.tight_layout()
        plt.savefig(f"{folder_path_Indentation_Depth}/{filename_noext}_CP_precise.png", bbox_inches='tight', dpi=300)
        plt.close('all') #To close all figures and save memory - max # of figures before warning: 20
    elif fitted_cp == 0:
        pass

    # 3. Indentation depth analysis
    x, y1, y2 = Time[:round(F_max_index+(F_max_index/10))]-Time[0], PiezoZ[:round(F_max_index+(F_max_index/10))], Force[:round(F_max_index+(F_max_index/10))]
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)', fontsize=18)
    ax1.set_ylabel('Piezo Z (um)', fontsize=18)
    lns1 = ax1.plot(x, y1, color='black', label='Piezo Z')
    ax1.tick_params(axis = 'y', labelcolor='black')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Force (uN)', fontsize=18, color='red')
    lns2 = ax2.plot(x, y2, color='red', label='Force')
    lns3 = ax2.axvline(x=Time[Est_CP_index]-Time[0], color='k', linestyle='--', label='Contact Point (approx)')
    lns4 = ax2.axvline(x=Time[F_max_index]-Time[0], color='b', linestyle='--', label='Maximum Force')
    lns5 = ax2.axvline(x=fitted_cp_time-Time[0], color='g', linestyle='--', label='Contact Point (Fitted)')
    ax2.tick_params(axis = 'y', labelcolor='red')

    ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.6))
    ax2.legend(loc='upper left', bbox_to_anchor=(0., 0.5))

    plt.yticks(fontsize=12)
    plt.title("{}".format(f"{filename_noext}_Indentation Depth Analysis"), fontsize=20)
    fig.tight_layout()
    plt.savefig(f"{folder_path_Indentation_Depth}/{filename_noext}_Indentation_Depth_Analysis.pdf", bbox_inches='tight')
    plt.close('all') #To close all figures and save memory - max # of figures before warning: 20

    return Displacement_Min_um, F_max_index, F_Min_uN, Max_Indentation_Depth_um, Ramp_Time, Force


def Fit_GenMaxwellModel_FixedT(path_filename, R, apply_rcf, correct_drift, v_sample):

    def GMM_fit(t, Ainf, A1, A2, A3, A4):
        return Ainf + (A1 * np.exp((-t)/t1)) + (A2 * np.exp((-t)/t2)) + (A3 * np.exp((-t)/t3)) + (A4 * np.exp((-t)/t4))

    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)
    folder_path_GMM_Fit = path + f"/GMM Fitting Results"

    # Convert raw txt data to df
    try:
        df = pd.read_table(path_filename, encoding="iso-8859-1", on_bad_lines='skip', low_memory=False,
                           delim_whitespace=True, names=(
                "Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]", "Pos Z [um]",
                "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]", "Gripper [V]",
                "Voltage A [V]",
                "Voltage B [V]", "Temperature [oC]", "Indent Depth [um]"))
        df = df[~df["Index [#]"].isin(
            ['//'])]  # Drop the rows that contain the comments to keep only the numbers    , encoding = "iso-8859-1"
        df = df.dropna(how='all')  # to drop if all values in the row are nan
        df = df.astype(float)  # Change data from object to float
    # Convert previously processed csv file to df
    except:
        df = pd.read_csv(path_filename)
        pass

    # Get contact point depth and max force index from data
    Displacement_Min_um, F_max_index, F_Min_uN, Max_Indentation_Depth_um, Ramp_Time, Force = DetermineIndentationDepth_F_relax(df, path, filename, correct_drift)

    # Get XY locations for array measurements
    PosX_um_col = df.columns.get_loc("Pos X [um]")  # Get column number
    PosY_um_col = df.columns.get_loc("Pos Y [um]")  # Get column number
    try:
        X = np.float64(df["Pos X [um]"].median())
        Y = np.float64(df["Pos Y [um]"].median())
    except IndexError:
        X = df.iloc[0, PosX_um_col]
        Y = df.iloc[0, PosY_um_col]

    # Get other columns required for data analysis
    Time_col = df.columns.get_loc("Time [s]")  # Get column number
    Force_col = df.columns.get_loc("Force A [uN]")  # Get column number
    PiezoZ_col = df.columns.get_loc("Piezo Z [um]")  # Get column number

    # Convert data to numpy arrays for processing
    Time = np.array(df.iloc[:, Time_col]) # define array of time data
    Force = np.array(df.iloc[:, Force_col])  # define array of force data
    PiezoZ = np.array(df.iloc[:, PiezoZ_col])  # define array of Piezo Z data

    # Determine point at which probe is retracted from sample (ie when the force relaxation experiment is complete)
    deltaPiezoZ = np.diff(PiezoZ) # Calculates difference between each adjacent value of PiezoZ
    MoveBack_Index = np.array([np.where(deltaPiezoZ < -0.01)]).min(initial=np._NoValue) # Find index of first time piezoZ moves away from sample. Originally -0.005

    # Select data for fitting to model between maximum force and move back index
    Length_Selected_Data = len(Time[F_max_index:MoveBack_Index])
    Sel_Time = Time[F_max_index:(MoveBack_Index-round(Length_Selected_Data/20))]-Time[F_max_index]
    Sel_Force_N = (Force[F_max_index:(MoveBack_Index-round(Length_Selected_Data/20))]-F_Min_uN)/1000000

    # Define parameters for fitting. tn could be changed to user parameter if needed
    F, t = Sel_Force_N, Sel_Time
    t1, t2, t3, t4 = 0.2, 2, 20, 200 # Alter time constants here if required

    # Fit data (default LM algorithm) to chosen model and extract optimal values for each parameter and standard deviations
    GMM_constants, GMM_Errors = curve_fit(GMM_fit, t, F, bounds=(0, 10000000), method='dogbox') #, p0=[400, 100, 100, 100, 100]  #Fit selected data to GMM model using least squares method , p0=[0.00001, 0.0001, 0.00001, 0.00001, 0.00001]
    Ainf_fit = GMM_constants[0]
    A1_fit = GMM_constants[1]
    A2_fit = GMM_constants[2]
    A3_fit = GMM_constants[3]
    A4_fit = GMM_constants[4]

    GMM_StDevs = np.sqrt(np.diag(GMM_Errors)) # Convert covariance to standard deviations

    Ainf_err = GMM_StDevs[0]
    A1_err = GMM_StDevs[1]
    A2_err = GMM_StDevs[2]
    A3_err = GMM_StDevs[3]
    A4_err = GMM_StDevs[4]

    #Determine force decay curve using fitted parameters
    GMM_Fit_F = [] #List of force calculated using derived fitting parameters (for plotting fitted curve)
    for i in t:
        GMM_Fit_F.append(GMM_fit(i, Ainf_fit, A1_fit, A2_fit, A3_fit, A4_fit)) #Calculates force at each value of time  (for plotting fitted curve)

    #Determine ramp correction factor (RCF) for each tn value
    RCF1 = (t1/Ramp_Time) * (np.exp(Ramp_Time/t1) - 1)
    RCF2 = (t2/Ramp_Time) * (np.exp(Ramp_Time/t2) - 1)
    RCF3 = (t3/Ramp_Time) * (np.exp(Ramp_Time/t3) - 1)
    RCF4 = (t4/Ramp_Time) * (np.exp(Ramp_Time/t4) - 1)

    #Convert parameter values from fitting to corresponding moduli and determine instantaneous modulus and viscoelastic ratio
    Max_Indentation_Depth_m = Max_Indentation_Depth_um / 1000000
    R_m = R / 1000000

    Eeq = 3/4 * (1 - v_sample**2) * (Ainf_fit / (Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5))
    E1 = 3/4 * (1 - v_sample**2) * (A1_fit / (RCF1 * Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5))
    E2 = 3/4 * (1 - v_sample**2)* (A2_fit / (RCF2 * Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5))
    E3 = 3/4 * (1 - v_sample**2)* (A3_fit / (RCF3 * Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5))
    E4 = 3/4 * (1 - v_sample**2) * (A4_fit / (RCF4 * Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5))
    Einst = Eeq + E1 + E2 + E3 + E4
    Viscoelastic_Ratio = Eeq / Einst

    Eeq_err = 3/4 * (1 - v_sample**2) * (Ainf_err / (Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5))
    E1_err = 3/4 * (1 - v_sample**2) * (A1_err / (RCF1 * Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5))
    E2_err = 3/4 * (1 - v_sample**2) * (A2_err / (RCF2 * Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5))
    E3_err = 3/4 * (1 - v_sample**2) * (A3_err / (RCF3 * Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5))
    E4_err = 3/4 * (1 - v_sample**2)* (A4_err / (RCF4 * Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5))
    Einst_err = Eeq_err + E1_err + E2_err + E3_err + E4_err
    Viscoelastic_Ratio_err = Viscoelastic_Ratio * ((Eeq_err / Eeq) + (Einst_err / Einst))


    #Print data in dialog box
    txt.insert(END, "Fitting completed:\n\n Equilibrium Modulus = {} [kPa], ".format(str(np.round(Eeq/1000, 1))))
    txt.insert(END, "Instantaneous Modulus = {} [kPa], ".format(str(np.round(Einst/1000, 1))))
    txt.insert(END, "Viscoelastic Ratio = {}. \n\n".format(str(np.round(Viscoelastic_Ratio, 2))))
    txt.update()
    txt.see("end")

    #Add calculated values to dataframe and save as csv
    df['Sel Time [s]'] = pd.Series(Sel_Time)
    df['Sel Force [N]'] = pd.Series(Sel_Force_N)
    df['Sel Force GMM Fit [N]'] = pd.Series(GMM_Fit_F)
    df.to_csv(f"{folder_path_GMM_Fit}/{filename_noext}_GMM Fit.csv", index=False)  # Save all measurements as *.csv, index=False avoids saving index

    #Calculate contact radius for plotting
    Contact_Radius_um = np.sqrt(Max_Indentation_Depth_um*R)

    # Data Plotting (Selected Data)
    x, y1, y2 = Sel_Time, Sel_Force_N, GMM_Fit_F
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)', fontsize=18)
    ax1.set_ylabel('Force (N)', fontsize=18)
    lns1 = ax1.plot(x, y1, color='black', label='Raw Data')
    lns2 = ax1.plot(x, y2, color='red', label='Fit')
    ax1.tick_params(axis='y', labelcolor='black')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    lns = lns1 + lns2 # To handle combined figure legend     +lns3
    labs = [l.get_label() for l in lns]  # To handle combined figure legend
    ax1.legend(lns, labs, fontsize=12)  # To handle combined figure legend
    plt.yticks(fontsize=12)
    plt.title("{}".format(f"{filename_noext}_GMM Fit"), fontsize=20)
    fig.tight_layout()
    plt.savefig(f"{folder_path_GMM_Fit}/{filename_noext}_GMM Fit.pdf", bbox_inches='tight')  # png ,dpi=300
    plt.close('all')  # To close all figures and save memory - max # of figures before warning: 20

    # GET MEASUREMENT TIME
    Measurement_Time = df.iloc[0,Time_col]
    txt.insert(END, "Measurement Time: {} s\n\n".format(Measurement_Time))
    txt.update()
    txt.see("end")

    #txt.insert(END, "Variables {}\n".format(GMM_constants))
    #txt.insert(END, "Errors {}\n".format(GMM_Errors))
    Spacer()

    return path, filename_noext, Eeq, Eeq_err, Einst, Einst_err, Viscoelastic_Ratio, Viscoelastic_Ratio_err, X, Y, Contact_Radius_um, Measurement_Time


def Fit_GenMaxwellModel(path_filename, R, v_sample, correct_drift, apply_rcf): #, threshold_constant

    def GMM_fit(t, Einf, E1, E2, E3, t1, t2, t3):
        return Einf + (E1 * np.exp((-t)/t1)) + (E2 * np.exp((-t)/t2)) + (E3 * np.exp((-t)/t3))

    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)
    folder_path_GMM_Fit = path + f"/GMM Fitting Results"

    # Convert raw txt data to df
    try:
        df = pd.read_table(path_filename, encoding="iso-8859-1", on_bad_lines='skip', low_memory=False,
                           delim_whitespace=True, names=(
                "Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]", "Pos Z [um]",
                "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]", "Gripper [V]",
                "Voltage A [V]",
                "Voltage B [V]", "Temperature [oC]", "Indent Depth [um]"))
        df = df[~df["Index [#]"].isin(
            ['//'])]  # Drop the rows that contain the comments to keep only the numbers    , encoding = "iso-8859-1"
        df = df.dropna(how='all')  # to drop if all values in the row are nan
        df = df.astype(float)  # Change data from object to float
    # Convert previously processed csv file to df
    except:
        df = pd.read_csv(path_filename)
        pass


    # Get other columns required for data analysis
    Time_col = df.columns.get_loc("Time [s]")  # Get column number
    Force_col = df.columns.get_loc("Force A [uN]")  # Get column number
    PiezoZ_col = df.columns.get_loc("Piezo Z [um]")  # Get column number
    Phase_col = df.columns.get_loc("Phase [#]")  # Get column number
    PosZ_col = df.columns.get_loc("Pos Z [um]")  # Get column number

    # Convert data to numpy arrays for processing
    Time = np.array(df.iloc[:, Time_col]) #define array of time data
    Force_uncor = np.array(df.iloc[:, Force_col])  #define array of force data
    PiezoZ = np.array(df.iloc[:, PiezoZ_col])  # define array of Piezo Z data
    Phase = np.array(df.iloc[:, Phase_col])
    PosZ = np.array(df.iloc[:, PosZ_col])


    #Get contact point depth and max force index from data
    Displacement_Min_um, F_max_index, F_Min_uN, Max_Indentation_Depth_um, Ramp_Time, Force = DetermineIndentationDepth_F_relax(df, path, filename, correct_drift)

    # Get XY locations for array measurements
    try:
        # The first few numbers in the X/Y columns occasionally (approx 1/100+ files) gets corrupted therefore use median X/Y
        X = np.float64(df["Pos X [um]"].median())
        Y = np.float64(df["Pos Y [um]"].median())
    except IndexError:
        PosX_um_col = df.columns.get_loc("Pos X [um]")  # Get column number
        PosY_um_col = df.columns.get_loc("Pos Y [um]")  # Get column number
        X = df.iloc[0, PosX_um_col]
        Y = df.iloc[0, PosY_um_col]

    # PREVIOUS METHOD OF MANUALLY DETERMINING PROBE RETRACTION POINT
    # Determine point at which probe is retracted from sample (ie when the force relaxation experiment is complete)
    # deltaPiezoZ = np.diff(PiezoZ) # Calculates difference between each adjacent value of PiezoZ
    # MoveBack_Index = np.array([np.where(deltaPiezoZ < -0.01)]).min() # Find index of first time piezoZ moves away from sample. Originally -0.005

    # NEW METHOD OF DETERMINING MEASUREMENT COMPLETION BY 'PHASE'
    MoveBack_Index = np.array([np.where(Phase >= 4)]).min(initial=np._NoValue)

    # OPTIONAL SECTION OF CODE TO REMOVE RELAXATION DATA WHERE ACTUATOR IS STILL MOVING
    # meanPiezoZ = np.mean(PiezoZ[F_max_index:MoveBack_Index])
    # sdPiezoZ = np.std(PiezoZ[F_max_index:MoveBack_Index])
    # PiezoZmax = meanPiezoZ + 100 * sdPiezoZ
    # txt.insert(END, "Mean PiezoZ = {}. \n\n".format(str(meanPiezoZ)))
    # txt.insert(END, "SD PiezoZ = {}. \n\n".format(str(sdPiezoZ)))
    # txt.insert(END, "Max PiezoZ = {}. \n\n".format(str(PiezoZmax)))
    # Index_Adjustment = np.array([np.where(PiezoZ[F_max_index:MoveBack_Index] < PiezoZmax)]).min()
    # txt.insert(END, "Fmax Index (pre adjustment) = {}. \n\n".format(str(F_max_index)))
    # F_max_index = F_max_index + Index_Adjustment
    # txt.insert(END, "Fmax Index (post adjustment) = {}. \n\n".format(str(F_max_index)))

    # Select data for fitting to model between maximum force and move back index
    Length_Selected_Data = len(Time[F_max_index:MoveBack_Index])
    Sel_Time = Time[F_max_index:(MoveBack_Index-round(Length_Selected_Data/20))]-Time[F_max_index]
    Sel_Force_N = (Force[F_max_index:(MoveBack_Index-round(Length_Selected_Data/20))]-F_Min_uN)/1000000
    Sel_Force_uncor_N = (Force_uncor[F_max_index:(MoveBack_Index-round(Length_Selected_Data/20))]-F_Min_uN)/1000000

    # Convert data for GMM fitting using form factor (i.e. convert force, F(t), to Young's Modulus, E(t), as a function of measurement time.
    Max_Indentation_Depth_m = Max_Indentation_Depth_um / 1000000
    R_m = R / 1000000
    contact_radius_m = np.sqrt(R_m * Max_Indentation_Depth_m)
    # Sel_E_t = Sel_Force_N * (7 * (1 + v_sample) * (1 - v_sample))/(4 * np.pi * R_m * Max_Indentation_Depth_m)  # Form factor obtained from Calixto et al. 2020 [mdpi.com/2073-4360/13/4/629] - gave unexpectedly low modulus values so replaced with Hertz model below.
    # Sel_E_t = Sel_Force_N / (16/9 * Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5)   # Form factor for Young's Modulus based on Hertz model with v = 0.5
    Sel_E_t = 3/4 * (1 - v_sample**2) * (Sel_Force_N / (Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5))    # Form factor for Young's Modulus based on Hertz model with variable v
    # Sel_E_uncor_t = Sel_Force_uncor_N / (16 / 9 * Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5)
    Sel_E_uncor_t = 3/4 * (1 - v_sample**2) * (Sel_Force_uncor_N / (Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5))

    # Define parameters for fitting. tn could be changed to user parameter if needed
    E, t = Sel_E_t, Sel_Time

    # Fit data (default LM algorithm) to chosen model and extract optimal values for each parameter and standard deviations
    GMM_constants, GMM_Errors = curve_fit(GMM_fit, t, E, bounds=([0.1, 0.1, 0.1, 0.1, 0.01, 0.51, 5.1], [10000000, 10000000, 10000000, 10000000, 10, 50, 10000]), p0=[500000, 10000, 10000, 10000, 0.1, 1, 10], maxfev=20000) #Fit selected data to GMM model using least squares method , p0=[0.00001, 0.0001, 0.00001, 0.00001, 0.00001]       method='dogbox',

    # Obtain fitted parameters and standard deviations from model
    Einf_fit, E1_fit, E2_fit, E3_fit, t1_fit, t2_fit, t3_fit = GMM_constants[0], GMM_constants[1], GMM_constants[2], GMM_constants[3], GMM_constants[4], GMM_constants[5], GMM_constants[6]
    GMM_StDevs = np.sqrt(np.diag(GMM_Errors)) # Convert covariance to standard deviations
    Einf_err, E1_err, E2_err, E3_err = GMM_StDevs[0], GMM_StDevs[1], GMM_StDevs[2], GMM_StDevs[3]

    # Predict force decay curve using fitted parameters
    GMM_Fit_E = [] # List of force calculated using derived fitting parameters (for plotting fitted curve)
    for i in t:
        GMM_Fit_E.append(GMM_fit(i, Einf_fit, E1_fit, E2_fit, E3_fit, t1_fit, t2_fit, t3_fit)) # Calculates force at each value of time  (for plotting fitted curve)
        
        # Applying 'Ramp Correction Factor' (see Oyen et al.) based on user checkbox
    if apply_rcf == 1:
        RCF1 = (t1_fit / Ramp_Time) * (np.exp(Ramp_Time / t1_fit) - 1)
        RCF2 = (t2_fit / Ramp_Time) * (np.exp(Ramp_Time / t2_fit) - 1)
        RCF3 = (t3_fit / Ramp_Time) * (np.exp(Ramp_Time / t3_fit) - 1)

        E1_fit = E1_fit / RCF1
        E2_fit = E2_fit / RCF2
        E3_fit = E3_fit / RCF3
        txt.insert(END,
                   "After RCF:\n\n Fitted Moduli [Pa]; E1 = {}, E2 = {}, E3 = {} \n".format(str(E1_fit), str(E2_fit),
                                                                                            str(E3_fit)))
        txt.insert(END, "Ramp time: {}s\n".format(str(Ramp_Time)))
    else:
        pass

    # Convert parameter values from fitting to corresponding moduli and determine instantaneous modulus and viscoelastic ratio
    Eeq = Einf_fit
    Einst = Eeq + E1_fit + E2_fit + E3_fit
    Viscoelastic_Ratio = Eeq / Einst
    Eeq_err = Einf_err
    Einst_err = Eeq_err + E1_err + E2_err + E3_err
    Viscoelastic_Ratio_err = Viscoelastic_Ratio * ((Eeq_err / Eeq) + (Einst_err / Einst))

    # Print data in dialog box
    txt.insert(END, "Fitting completed:\n\n Time constants [s]; t1 = {}, t2 = {}, t3 = {} \n".format(str(t1_fit), str(t2_fit), str(t3_fit)))
    txt.insert(END, "Fitting completed:\n\n Fitted Moduli [Pa]; E1 = {}, E2 = {}, E3 = {} \n".format(str(E1_fit), str(E2_fit), str(E3_fit)))
    txt.insert(END, "Equilibrium Modulus = {} [kPa], ".format(str(np.round(Eeq/1000, 1))))
    txt.insert(END, "Instantaneous Modulus = {} [kPa], ".format(str(np.round(Einst/1000, 1))))
    txt.insert(END, "Viscoelastic Ratio = {}. \n\n".format(str(np.round(Viscoelastic_Ratio, 2))))
    txt.update()
    txt.see("end")

    Finf = (Einf_fit * 4 * Max_Indentation_Depth_m ** 1.5 * R_m ** 0.5) / (3 * (1 - v_sample**2))
    Sel_Force_N_norm = (Sel_Force_N-Finf)/(Sel_Force_N[0]-Finf)
    Sel_Time_norm = Sel_Time/(R_m*Max_Indentation_Depth_m) ** 2

    # Add calculated values to dataframe and save as csv
    df['Sel Time [s]'] = pd.Series(Sel_Time)
    df['Sel Force [N]'] = pd.Series(Sel_Force_N)
    df['Sel Modulus Uncorrected [Pa]'] = pd.Series(Sel_E_uncor_t)
    df['Sel Modulus Corrected [Pa]'] = pd.Series(Sel_E_t)
    df['Sel Modulus GMM Fit [Pa]'] = pd.Series(GMM_Fit_E)
    df['Sel Time Normalised [s m-2]'] = pd.Series(Sel_Time_norm)
    df['Sel Force Normalised []'] = pd.Series(Sel_Force_N_norm)

    df.to_csv(f"{folder_path_GMM_Fit}/{filename_noext}_GMM Fit.csv", index=False)  # Save all measurements as *.csv, index=False avoids saving index

    # Calculate contact radius for plotting
    Contact_Radius_um = np.sqrt(Max_Indentation_Depth_um*R)

    # Data Plotting (Selected Data)
    global lns
    x, y1, y2, y3 = Sel_Time, Sel_E_t, GMM_Fit_E, Sel_E_uncor_t
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)', fontsize=18)
    ax1.set_ylabel('Modulus (Pa)', fontsize=18)
    ax1.tick_params(axis='y', labelcolor='black')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if correct_drift == 1:
        # lns3 = ax1.plot(x, y3, color='blue', label='Uncorrected Data')
        # lns1 = ax1.plot(x, y1, color='black', label='Corrected Data')
        lns3 = ax1.plot(x, y3, '.', color='lightsteelblue', label='Uncorrected Data')
        lns1 = ax1.plot(x, y1, '.', color='steelblue', label='Corrected Data')
        lns2 = ax1.plot(x, y2, color='black', label='GMM Fit')
        lns = lns3 + lns1 + lns2 # To handle combined figure legend    , 'ko'
    elif correct_drift == 0:
        lns1 = ax1.plot(x, y1, '.', color='steelblue', label='Raw Data')
        lns2 = ax1.plot(x, y2, color='black', label='GMM Fit')
        lns = lns1 + lns2 # To handle combined figure legend
    labs = [l.get_label() for l in lns]  # To handle combined figure legend
    ax1.legend(lns, labs, fontsize=12)  # To handle combined figure legend
    plt.yticks(fontsize=12)
    plt.title("{}".format(f"{filename_noext}_GMM Fit"), fontsize=20)
    fig.tight_layout()
    plt.savefig(f"{folder_path_GMM_Fit}/{filename_noext}_GMM Fit.png", bbox_inches='tight')  # png ,dpi=300
    plt.close('all')  # To close all figures and save memory - max # of figures before warning: 20

    # Fourier transform to calculate storage and loss moduli
    omega = np.geomspace(0.01, 500, 100)
    Ei = Einf_fit + (E1_fit * t1_fit ** 2 * omega ** 2)/(1 + t1_fit ** 2 * omega ** 2) + (E2_fit * t2_fit ** 2 * omega ** 2)/(1 + t2_fit ** 2 * omega ** 2) + (E3_fit * t3_fit ** 2 * omega ** 2)/(1 + t3_fit ** 2 * omega ** 2)
    Eii = (E1_fit * t1_fit * omega)/(1 + t1_fit ** 2 * omega ** 2) + (E2_fit * t2_fit * omega)/(1 + t2_fit ** 2 * omega ** 2) + (E3_fit * t3_fit * omega)/(1 + t3_fit ** 2 * omega ** 2)
    E_dynamic = np.sqrt(Ei ** 2 + Eii ** 2)
    TanDelta = Eii/Ei

    Ei_1hz = Einf_fit + (E1_fit * t1_fit ** 2 * 1 ** 2)/(1 + t1_fit ** 2 * 1 ** 2) + (E2_fit * t2_fit ** 2 * 1 ** 2)/(1 + t2_fit ** 2 * 1 ** 2) + (E3_fit * t3_fit ** 2 * 1 ** 2)/(1 + t3_fit ** 2 * 1 ** 2)
    Ei_10hz = Einf_fit + (E1_fit * t1_fit ** 2 * 10 ** 2)/(1 + t1_fit ** 2 * 10 ** 2) + (E2_fit * t2_fit ** 2 * 10 ** 2)/(1 + t2_fit ** 2 * 10 ** 2) + (E3_fit * t3_fit ** 2 * 10 ** 2)/(1 + t3_fit ** 2 * 10 ** 2)
    Eii_1hz = (E1_fit * t1_fit * 1)/(1 + t1_fit ** 2 * 1 ** 2) + (E2_fit * t2_fit * 1)/(1 + t2_fit ** 2 * 1 ** 2) + (E3_fit * t3_fit * 1)/(1 + t3_fit ** 2 * 1 ** 2)
    Eii_10hz = (E1_fit * t1_fit * 10)/(1 + t1_fit ** 2 * 10 ** 2) + (E2_fit * t2_fit * 10)/(1 + t2_fit ** 2 * 10 ** 2) + (E3_fit * t3_fit * 10)/(1 + t3_fit ** 2 * 10 ** 2)
    TanDelta_1hz = Eii_1hz/Ei_1hz
    TanDelta_10hz = Eii_10hz/Ei_10hz

    # Plot storage/loss moduli as function of frequency
    x, y1, y2 = omega, Ei, Eii
    fig, ax1 = plt.subplots()
    plt.yscale("log")
    plt.xscale("log")

    ax1.set_xlabel('Frequency (Hz)', fontsize=18)
    ax1.set_ylabel('Modulus (Pa)', fontsize=18)
    lns1 = ax1.plot(x, y1, color='black', label='Storage Modulus')
    lns2 = ax1.plot(x, y2, color='red', label='Loss Modulus')
    ax1.tick_params(axis='y', labelcolor='black')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    lns = lns1 + lns2 # To handle combined figure legend     +lns3
    labs = [l.get_label() for l in lns]  # To handle combined figure legend
    ax1.legend(lns, labs, fontsize=12)  # To handle combined figure legend
    plt.yticks(fontsize=12)
    plt.title("{}".format(f"{filename_noext}_Dynamic Moduli"), fontsize=20)
    fig.tight_layout()
    plt.savefig(f"{folder_path_GMM_Fit}/{filename_noext}_Dynamic Moduli.pdf", bbox_inches='tight')  # png ,dpi=300
    plt.close('all')  # To close all figures and save memory - max # of figures before warning: 20

    # GET MEASUREMENT TIME
    Measurement_Time = df.iloc[0,Time_col]
    txt.insert(END, "Measurement Time: {} s\n\n".format(Measurement_Time))
    txt.update()
    txt.see("end")
    Spacer()

    # GET HEIGHT DATA
    PosZ_CP = np.float64(df["Pos Z [um]"].median())
    PiezoZ_CP = Displacement_Min_um

    return path, filename_noext, Eeq, Eeq_err, Einst, Einst_err, Viscoelastic_Ratio, Viscoelastic_Ratio_err, X, Y, Contact_Radius_um, Measurement_Time, Ei, Eii, TanDelta, Ei_1hz, Eii_1hz, Ei_10hz, Eii_10hz, TanDelta_1hz, TanDelta_10hz, t1_fit, t2_fit, t3_fit, PosZ_CP, PiezoZ_CP, E1_fit, E2_fit, E3_fit


def Fit_PoroelasticModel(path_filename, R, correct_drift): #, threshold_constant

    def Poroelastic_fit(t, Pinf, P0, D):
        return Pinf + (P0 - Pinf) * ((0.491 * np.exp(-0.908 * ((D * t / (a0 ** 2)) ** 0.5))) + (0.509 * np.exp(-1.679 * ((D * t / (a0 ** 2)) ** 0.5))))

    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)
    folder_path_Poroelastic_Fit = path + f"/Poroelastic Fitting Results"
    #Create_Output_Folder(folder_path_GMM_Fit)

    #Convert raw txt data to df
    try:
        df = pd.read_table(path_filename, encoding="iso-8859-1", on_bad_lines='skip', low_memory=False,
                           delim_whitespace=True, names=(
                "Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]", "Pos Z [um]",
                "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]", "Gripper [V]",
                "Voltage A [V]",
                "Voltage B [V]", "Temperature [oC]", "Indent Depth [um]"))
        df = df[~df["Index [#]"].isin(
            ['//'])]  # Drop the rows that contain the comments to keep only the numbers    , encoding = "iso-8859-1"
        df = df.dropna(how='all')  # to drop if all values in the row are nan
        df = df.astype(float)  # Change data from object to float
    # Convert previously processed csv file to df
    except:
        df = pd.read_csv(path_filename)
        pass

    # Get contact point depth and max force index from data
    Displacement_Min_um, F_max_index, F_Min_uN, Max_Indentation_Depth_um, Ramp_Time, Force = DetermineIndentationDepth_F_relax(df, path, filename, correct_drift)

    # Get XY locations for array measurements
    PosX_um_col = df.columns.get_loc("Pos X [um]")  # Get column number
    PosY_um_col = df.columns.get_loc("Pos Y [um]")  # Get column number
    try:
        X = np.float64(df["Pos X [um]"].median())
        Y = np.float64(df["Pos Y [um]"].median())
    except IndexError:
        X = df.iloc[0, PosX_um_col]
        Y = df.iloc[0, PosY_um_col]

    # Get other columns required for data analysis
    Time_col = df.columns.get_loc("Time [s]")  # Get column number
    Force_col = df.columns.get_loc("Force A [uN]")  # Get column number
    PiezoZ_col = df.columns.get_loc("Piezo Z [um]")  # Get column number
    Phase_col = df.columns.get_loc("Phase [#]")  # Get column number

    # Convert data to numpy arrays for processing
    Time = np.array(df.iloc[:, Time_col]) # define array of time data
    Force = np.array(df.iloc[:, Force_col])  # define array of force data
    PiezoZ = np.array(df.iloc[:, PiezoZ_col])  # define array of Piezo Z data
    Phase = np.array(df.iloc[:, Phase_col])

    MoveBack_Index = np.array([np.where(Phase >= 4)]).min(initial=np._NoValue)

    # Index adjustment
    meanPiezoZ = np.mean(PiezoZ[F_max_index:MoveBack_Index])
    sdPiezoZ = np.std(PiezoZ[F_max_index:MoveBack_Index])
    PiezoZmax = meanPiezoZ + 100 * sdPiezoZ
    txt.insert(END, "Mean PiezoZ = {}. \n\n".format(str(meanPiezoZ)))
    txt.insert(END, "SD PiezoZ = {}. \n\n".format(str(sdPiezoZ)))
    txt.insert(END, "Max PiezoZ = {}. \n\n".format(str(PiezoZmax)))

    Index_Adjustment = np.array([np.where(PiezoZ[F_max_index:MoveBack_Index] < PiezoZmax)]).min(initial=np._NoValue)
    txt.insert(END, "Fmax Index (pre adjustment) = {}. \n\n".format(str(F_max_index)))
    F_max_index = F_max_index + Index_Adjustment
    txt.insert(END, "Fmax Index (post adjustment) = {}. \n\n".format(str(F_max_index)))

    #Determine point at which probe is retracted from sample (ie when the force relaxation experiment is complete)
    #deltaPiezoZ = np.diff(PiezoZ) # Calculates difference between each adjacent value of PiezoZ
    #MoveBack_Index = np.array([np.where(deltaPiezoZ < -0.01)]).min() # Find index of first time piezoZ moves away from sample. Originally -0.005 but gave unreliable results

    #Select data for fitting to model between maximum force and move back index
    Length_Selected_Data = len(Time[F_max_index:MoveBack_Index])
    Sel_Time = Time[F_max_index:(MoveBack_Index-round(Length_Selected_Data/20))]-Time[F_max_index]
    Sel_Force_N = (Force[F_max_index:(MoveBack_Index-round(Length_Selected_Data/20))]-F_Min_uN)/1000000

    # Define parameters for fitting.
    F, t = Sel_Force_N, Sel_Time
    R_m = R / 1000000
    Max_Indentation_Depth_m = Max_Indentation_Depth_um / 1000000
    a0 = (Max_Indentation_Depth_m * R_m) ** 0.5

    # Fit data (default LM algorithm) to chosen model and extract optimal values for each parameter and standard deviations
    Poroelastic_constants, Poroelastic_Errors = curve_fit(Poroelastic_fit, t, F, p0=[min(F), max(F), 0.0000000000000001], method='lm', maxfev=100000) #, p0=[400, 100, 100, 100, 100]  #Fit selected data to GMM model using least squares method , p0=[0.00001, 0.0001, 0.00001, 0.00001, 0.00001] bounds=(0, 10000000), , bounds=(0, 100000)

    Pinf_fit, P0_fit, D_fit = Poroelastic_constants[0], Poroelastic_constants[1], Poroelastic_constants[2]
    Poroelastic_StDevs = np.sqrt(np.diag(Poroelastic_Errors)) # Convert covariance to standard deviations
    Pinf_err, P0_err, D_err = Poroelastic_StDevs[0], Poroelastic_StDevs[1], Poroelastic_StDevs[2]

    # Determine force decay curve using fitted parameters
    Poroelastic_Fit_F = [] #List of force calculated using derived fitting parameters (for plotting fitted curve)
    for i in t:
        Poroelastic_Fit_F.append(Poroelastic_fit(i, Pinf_fit, P0_fit, D_fit)) #Calculates force at each value of time  (for plotting fitted curve)

    # Determine other poroelastic parameters
    v_sample = (2 - P0_fit/Pinf_fit) / 2
    shear_modulus_Pa = P0_fit * 3/(16 * a0 * Max_Indentation_Depth_m)
    solvent_viscosity = 0.89 * 10 ** -3 # Pa s-1 (value for water - change if needed)
    instrinsic_permeability = (D_fit * (1 - 2 * v_sample) * solvent_viscosity)/(2 * (1 - v_sample) * shear_modulus_Pa)

    # Print data in dialog box
    txt.insert(END, "Fitting completed:\n\n Equilibrium Force = {} [uN], ".format(str(round(Pinf_fit*1000000, 1))))
    txt.insert(END, "Instantaneous Force = {} [uN], ".format(str(round(P0_fit*1000000, 1))))
    txt.insert(END, "Effective Diffusivity = {} [m2s-1]. \n\n".format(str(D_fit)))
    txt.insert(END, "Sample Poisson's ratio = {}. \n\n".format(str(v_sample)))
    txt.insert(END, "Shear Modulus = {} [Pa]. \n\n".format(str(shear_modulus_Pa)))
    txt.insert(END, "Intrinsic Permeability = {} [m2]. \n\n".format(str(instrinsic_permeability)))
    txt.update()
    txt.see("end")

    # Add calculated values to dataframe and save as csv
    df['Sel Time'] = pd.Series(Sel_Time)
    df['Sel Force N'] = pd.Series(Sel_Force_N)
    df.to_csv(f"{folder_path_Poroelastic_Fit}/{filename_noext}_Poroelastic Fit.csv", index=False)  # Save all measurements as *.csv, index=False avoids saving index

    # Calculate contact radius for plotting
    Contact_Radius_um = np.sqrt(Max_Indentation_Depth_um*R)

    # Data Plotting (Selected Data)
    x, y1, y2 = Sel_Time, Sel_Force_N, Poroelastic_Fit_F
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)', fontsize=18)
    ax1.set_ylabel('Force (N)', fontsize=18)
    lns1 = ax1.plot(x, y1, color='black', label='Raw Data')
    lns2 = ax1.plot(x, y2, color='red', label='Fit')
    ax1.tick_params(axis='y', labelcolor='black')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    lns = lns1 + lns2 # To handle combined figure legend     +lns3
    labs = [l.get_label() for l in lns]  # To handle combined figure legend
    ax1.legend(lns, labs, fontsize=12)  # To handle combined figure legend
    plt.yticks(fontsize=12)
    plt.title("{}".format(f"{filename_noext}_Poroelastic Fit"), fontsize=20)
    fig.tight_layout()
    plt.savefig(f"{folder_path_Poroelastic_Fit}/{filename_noext}_Poroelastic Fit.pdf", bbox_inches='tight')  # png ,dpi=300
    plt.close('all')  # To close all figures and save memory - max # of figures before warning: 20

    # GET MEASUREMENT TIME
    Measurement_Time = df.iloc[0,Time_col]
    txt.insert(END, "Measurement Time: {} s\n\n".format(Measurement_Time))
    txt.update()
    txt.see("end")
    Spacer()

    # GET HEIGHT DATA
    PosZ_CP = np.float64(df["Pos Z [um]"].median())
    PiezoZ_CP = Displacement_Min_um
    Spacer()

    return path, filename_noext, Pinf_fit, Pinf_err, P0_fit, P0_err, D_fit, D_err, X, Y, Contact_Radius_um, Measurement_Time, PosZ_CP, PiezoZ_CP

# ----------------------------------------------------------------------------------------------------------------------#
#GUI
# ----------------------------------------------------------------------------------------------------------------------#

#Run Software
root  = Tk()
root.wm_title("Gobbo Group – A.L.I.A.S. - Viscoelastic") #A.L.I.A.S. = A Lovely Indentation Analysis System
root.geometry("1350x650")

#SET VARIABLES TO BE USED IN OTHER BUTTONS
#NOTE: Tkinter button cannot return a variable. To modify these variables inside the button's function and make them available outside
# the button's function you need to declare them outside the function and then declare them as global inside the function.
path_filename = ""
path = ""
filename = ""
filename_noext = ""
extension = ""
folder_path = ""
path_foldername = ""

year_now = datetime.now().strftime('%Y')
month_now = datetime.now().strftime("%B")  # returns the full name of the month as a string


# SOFTWARE FUNCTIONS
def File_Extractor_b1():
    """Button to carry out extraction of data.txt files from within a folder of indentation experiments."""
    dir = filedialog.askdirectory(title="Select a folder")
    # to handle Cancel button
    if dir == "":
        return

    txt.insert(END, f"Working directory: {dir}\n\n")
    txt.update()
    txt.see("end")

    #Find all folders within directory
    folder_list = glob.glob(f"{dir}/*/")

    counter_found = 0
    for folder_path_name in folder_list:
        elements = folder_path_name.split("\\")
        folder_name = elements[len(elements)-2]
        txt.insert(END, f"Extracting from: {folder_name}\n")
        txt.update()
        txt.see("end")

        found = 0
        for root, dirs, files in os.walk(folder_path_name, topdown=False):
            for name in files:
                # print(os.path.join(root, name))
                if name == "data.txt":
                    if os.path.isfile(f"{dir}/{folder_name}.txt") == True:
                        shutil.copy(f"{os.path.join(root, name)}", f"{dir}/{folder_name}_{counter_found+1}.txt")
                    else:
                        shutil.copy(f"{os.path.join(root, name)}", f"{dir}/{folder_name}.txt")
                    txt.insert(END, f"File found: {name}.\n")
                    txt.update()
                    txt.see("end")
                    counter_found = counter_found+1
                    found = 1

        if found == 0:
            txt.insert(END, f"Error: No file data.txt found.\n")
            txt.update()
            txt.see("end")

        txt.insert(END, f"Moving {folder_name} to backup folder 'Original data'\n")
        txt.insert(END, f"Operation successful!\n\n")
        txt.update()
        txt.see("end")
        shutil.copytree(f"{folder_path_name[:-1]}", f"{dir}/Original data/{folder_name}")
        shutil.rmtree(f"{folder_path_name[:-1]}")

    txt.insert(END, f"Found {counter_found} data.txt files in {len(folder_list)} experiment folders.\n\n")
    txt.update()
    txt.see("end")
    Spacer()
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")


def ForceRelaxationAnalysis_Individual_b2():
    """Button to carry out analysis of an array of force relaxation measurements.
    Inputs: probe diameter, Poisson's Ratio, model.
    Options: correct force drift, apply ramp correction faction (RCF).
    Outputs: csv files with output data from model, individual plots of contact point determination and model fitting."""

    # Set variables required
    global path_filename, path, filename, filename_noext, extension

    model = clicked.get()
    R = (float(e1.get())) / 2
    path_filenames = filedialog.askopenfilenames(title="Select all files to process", filetypes = [("TXT files", "*.txt")])
    v_sample = float(e2.get())
    correct_drift = var1.get()
    apply_rcf = var2.get()

    if path_filenames == "":
        return

    txt.insert(END, f"Analysing selected txt files...\n")
    txt.update()
    txt.see("end")
    txt.insert(END, "Number of files to process: {}\n\n\n".format(len(path_filenames)))
    txt.update()
    txt.see("end")

    if model == "GMM - Fixed Decay Times":
        Spacer()
        Spacer()
        txt.insert(END, "Fitting data to Generalised Maxwell Model...\n")
        txt.update()
        txt.see("end")
        Spacer()

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)
            Create_Output_Folder(path + f"/Indentation_Depth_Analysis")

            #GMM Fitting
            Create_Output_Folder(path + f"/GMM Fitting Results")
            break #Prevents repeated attempts to create same folder

        Labels_list = ['File name', 'Measurement Time [s]', 'X [um]', 'Y [um]', 'Equilibrium Modulus [kPa]', 'StDev [kPa]', 'Instantaneous Modulus [kPa]', 'StDev [kPa]', 'Viscoelastic Ratio', 'StDev'] #List of column headings for results csv file
        Results_list = [] #List of determined parameters for results csv file

        for path_filename in path_filenames:
            path, filename_noext, Eeq, Eeq_err, Einst, Einst_err, Viscoelastic_Ratio, Viscoelastic_Ratio_err, X, Y, Contact_Radius_um, Measurement_Time = Fit_GenMaxwellModel_FixedT(path_filename, R, apply_rcf, correct_drift, v_sample)
            Results_list.append([filename_noext, Measurement_Time, X, Y, Eeq/1000, Eeq_err/1000, Einst/1000, Einst_err/1000, Viscoelastic_Ratio, Viscoelastic_Ratio_err])

        with open("{}/Fitting Results Summary_GMM.csv".format(path), 'w') as f:
            write = csv.writer(f)
            write.writerow(Labels_list)
            write.writerows(Results_list)

        Spacer()
        txt.insert(END, "Summary saved in: {}/Fitting Results Summary_GMM.csv\n".format(path))

    if model == "GMM":
        Spacer()
        Spacer()
        txt.insert(END, "Fitting data to Generalised Maxwell Model...\n")
        txt.update()
        txt.see("end")
        Spacer()

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)
            Create_Output_Folder(path + f"/Indentation_Depth_Analysis")

            #GMM Fitting
            Create_Output_Folder(path + f"/GMM Fitting Results")
            break #Prevents repeated attempts to create same folder

        #Labels_list = ['File name', 'Measurement Time [s]', 'X [um]', 'Y [um]', 'Equilibrium Modulus [kPa]', 'StDev [kPa]', 'Instantaneous Modulus [kPa]', 'StDev [kPa]', 'Viscoelastic Ratio', 'StDev'] #List of column headings for results csv file
        Labels_list = ['File name', 'Measurement Time [s]', 'X [um]', 'Y [um]', 'Equilibrium Modulus [kPa]', 'StDev [kPa]', 'Instantaneous Modulus [kPa]', 'StDev [kPa]', 'Viscoelastic Ratio', 'StDev', 'Storage Modulus 1Hz [kPa]', 'Loss Modulus 1Hz [kPa]', 'Tan Delta 1Hz', 'Storage Modulus 10Hz [kPa]', 'Loss Modulus 10Hz [kPa]', 'Tan Delta 10Hz','t1 [s]', 't2 [s]', 't3 [s]', 'PosZ_CP [um]', 'PiezoZ_CP [um]', 'E1 [kPa]', 'E2 [kPa]', 'E3 [kPa]', 'E1 Relative', 'E2 Relative', 'E3 Relative', 'Indentation depth [um]']  # List of column headings for results csv file
        Results_list = [] #List of determined parameters for results csv file

        X_list = []
        Y_list = []

        omega = np.geomspace(0.01, 500, 100)
        omega_list = list(omega)
        omega_list.insert(0, 'Filename')
        StorageModulus_df = pd.DataFrame(columns=omega_list)
        LossModulus_df = pd.DataFrame(columns=omega_list)
        TanDelta_df = pd.DataFrame(columns=omega_list)

        for path_filename in path_filenames:
            # path, filename_noext, Eeq, Eeq_err, Einst, Einst_err, Viscoelastic_Ratio, Viscoelastic_Ratio_err, X, Y, Contact_Radius_um, Measurement_Time, Ei, Eii, TanDelta, Ei_1hz, Eii_1hz, Ei_10hz, Eii_10hz, TanDelta_1hz, TanDelta_10hz = Fit_GenMaxwellModel_test(
            #     path_filename, R, v_sample)
            try:
                df = pd.read_table(path_filename, encoding="iso-8859-1", on_bad_lines='skip', low_memory=False,
                                   delim_whitespace=True, names=(
                        "Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]",
                        "Pos Z [um]",
                        "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]", "Gripper [V]",
                        "Voltage A [V]",
                        "Voltage B [V]", "Temperature [oC]", "Indent Depth [um]"))
                df = df[~df["Index [#]"].isin(
                    [
                        '//'])]  # Drop the rows that contain the comments to keep only the numbers    , encoding = "iso-8859-1"
                df = df.dropna(how='all')  # to drop if all values in the row are nan
                df = df.astype(float)  # Change data from object to float
            # Convert previously processed csv file to df
            except:
                df = pd.read_csv(path_filename)
                pass
            Displacement_Min_um, F_max_index, F_Min_uN, Max_Indentation_Depth_um, Ramp_Time, Force = DetermineIndentationDepth_F_relax(df, path, filename, correct_drift)

            try:
                path, filename_noext, Eeq, Eeq_err, Einst, Einst_err, Viscoelastic_Ratio, Viscoelastic_Ratio_err, X, Y, Contact_Radius_um, Measurement_Time, Ei, Eii, TanDelta, Ei_1hz, Eii_1hz, Ei_10hz, Eii_10hz, TanDelta_1hz, TanDelta_10hz, t1, t2, t3, PosZ_CP, PiezoZ_CP, E1, E2, E3 = Fit_GenMaxwellModel(path_filename, R, v_sample, correct_drift, apply_rcf)


            except (ValueError, np.linalg.LinAlgError, RuntimeError, StopIteration, IndexError): #ValueError, np.linalg.LinAlgError, RuntimeError, StopIteration, IndexError
                try:
                    df = pd.read_table(path_filename, encoding="iso-8859-1", on_bad_lines='skip', low_memory=False,
                                           delim_whitespace=True, names=(
                        "Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]",
                        "Pos Z [um]", "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]",
                        "Gripper [V]", "Voltage A [V]", "Voltage B [V]", "Temperature [oC]"))
                    df = df[~df["Index [#]"].isin([
                                                      '//'])]  # Drop the rows that contain the comments to keep only the numbers    , encoding = "iso-8859-1"
                    df = df.dropna(how='all')  # to drop if all values in the row are nan
                    df = df.astype(float)  # Change data from object to float
                # Convert previously processed csv file to df
                except:
                    df = pd.read_csv(path_filename)
                    pass

                # Get XY locations for array measurements
                PosX_um_col = df.columns.get_loc("Pos X [um]")  # Get column number
                PosY_um_col = df.columns.get_loc("Pos Y [um]")  # Get column number
                try:
                    X = np.float64(df["Pos X [um]"].median())
                    Y = np.float64(df["Pos Y [um]"].median())

                except IndexError:
                    X = df.iloc[0, PosX_um_col]
                    Y = df.iloc[0, PosY_um_col]

                filename_noext, Eeq, Eeq_err, Einst, Einst_err, Viscoelastic_Ratio, Viscoelastic_Ratio_err, Contact_Radius_um, Measurement_Time, Ei_1hz, Eii_1hz, Ei_10hz, Eii_10hz, TanDelta_1hz, TanDelta_10hz = filename, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            #path, filename_noext, Eeq, Eeq_err, Einst, Einst_err, Viscoelastic_Ratio, Viscoelastic_Ratio_err, X, Y, Contact_Radius_um, Measurement_Time, Ei, Eii, TanDelta, Ei_1hz, Eii_1hz, Ei_10hz, Eii_10hz, TanDelta_1hz, TanDelta_10hz, t1, t2, t3, PosZ_CP, PiezoZ_CP, E1, E2, E3 = Fit_GenMaxwellModel(path_filename, R, v_sample, correct_drift, apply_rcf)

            #Results_list.append([filename_noext, Measurement_Time, X, Y, Eeq/1000, Eeq_err/1000, Einst/1000, Einst_err/1000, Viscoelastic_Ratio, Viscoelastic_Ratio_err])
            Results_list.append(
                [filename_noext, Measurement_Time, X, Y, Eeq / 1000, Eeq_err / 1000, Einst / 1000, Einst_err / 1000,
                 Viscoelastic_Ratio, Viscoelastic_Ratio_err, Ei_1hz / 1000, Eii_1hz / 1000, TanDelta_1hz, Ei_10hz / 1000, Eii_10hz / 1000, TanDelta_10hz, t1, t2, t3, PosZ_CP, PiezoZ_CP, E1 / 1000, E2 / 1000, E3 / 1000, E1 / Einst, E2 / Einst,  E3 / Einst, Max_Indentation_Depth_um])
            X_list.append(X)
            Y_list.append(Y)

            Ei = list(Ei)
            Eii = list(Eii)
            TanDelta = list(TanDelta)

            Ei.insert(0, filename_noext)
            Eii.insert(0, filename_noext)
            TanDelta.insert(0, filename_noext)

            StorageModulus_df.loc[len(StorageModulus_df)] = pd.Series(Ei, index=omega_list)
            LossModulus_df.loc[len(LossModulus_df)] = pd.Series(Eii, index=omega_list)
            TanDelta_df.loc[len(TanDelta_df)] = pd.Series(TanDelta, index=omega_list)

        # Summarise (mean and st dev for each frequency) dynamic moduli data
        Ei_df_nofilenames = StorageModulus_df.iloc[:, 1:]
        SM_means = Ei_df_nofilenames.mean(axis=0)
        #SM_stdevs = Ei_df_nofilenames.std(axis=0)

        Eii_df_nofilenames = LossModulus_df.iloc[:, 1:]
        LM_means = Eii_df_nofilenames.mean(axis=0)
        #LM_stdevs = Eii_df_nofilenames.std(axis=0)

        TD_df_nofilenames = TanDelta_df.iloc[:, 1:]
        TD_means = TD_df_nofilenames.mean(axis=0)
        #TD_stdevs = TD_df_nofilenames.std(axis=0)

        x = omega
        #y1, y2, y3 = (SM_means, SM_stdevs), (LM_means, LM_stdevs), (TD_means, TD_stdevs)
        y1, y2, y3 = SM_means, LM_means, TD_means

        with open("{}/Dynamic Moduli Summary_GMM.csv".format(path), 'w', newline='') as f: #newline='' is necessary otherwise it creates an empty line between each record.
            write = csv.writer(f)
            Dynamic_moduli_labels = ['Frequency', 'Storage Modulus [Pa]', 'Loss Modulus [Pa]', 'Tan Delta']
            DM_list = [x, y1, y2, y3]
            write.writerow(Dynamic_moduli_labels)
            write.writerows(zip(*DM_list))
        Spacer()
        txt.insert(END, "Summary saved in: {}/Dynamic Moduli Summary_GMM.csv\n".format(path))

        with open("{}/Fitting Results Summary_GMM.csv".format(path), 'w') as f:
            write = csv.writer(f)
            write.writerow(Labels_list)
            write.writerows(Results_list)

        Spacer()
        txt.insert(END, "Summary saved in: {}/Fitting Results Summary_GMM.csv\n".format(path))

    if model == "Poroelastic":
        Spacer()
        Spacer()
        txt.insert(END, "Fitting data to Poroelastic Model...\n")
        txt.update()
        txt.see("end")
        Spacer()

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)
            Create_Output_Folder(path + f"/Indentation_Depth_Analysis")

            # GMM Fitting
            Create_Output_Folder(path + f"/Poroelastic Fitting Results")
            break  # Prevents repeated attempts to create same folder

        Labels_list = ['File name', 'Measurement Time [s]', 'X [um]', 'Y [um]', 'Equilibrium Force [uN]', 'StDev [uN]', 'Instantaneous Force [uN]', 'StDev [uN]', 'Effective Diffusivity [m2s-1]', 'StDev [m2s-1]']  # List of column headings for results csv file
        Results_list = []  # List of determined parameters for results csv file

        for path_filename in path_filenames:
            path, filename_noext, Pinf_fit, Pinf_err, P0_fit, P0_err, D_fit, D_err, X, Y, Contact_Radius_um, Measurement_Time, PosZ_CP, PiezoZ_CP = Fit_PoroelasticModel(path_filename, R, correct_drift)
            Results_list.append([filename_noext, Measurement_Time, X, Y, Pinf_fit*1000000, Pinf_err*1000000, P0_fit*1000000, P0_err*1000000, D_fit, D_err])

        with open("{}/Fitting Results Summary_Poroelastic.csv".format(path), 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(Labels_list)
            write.writerows(Results_list)

        Spacer()
        txt.insert(END, "Summary saved in: {}/Fitting Results Summary_Poroelastic.csv\n".format(path))

    Spacer()
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")


def ForceRelaxationAnalysis_Array_b3():
    """Button to carry out analysis of an array of force relaxation measurements.
    Inputs: probe diameter, Poisson's Ratio, model.
    Options: correct force drift, apply ramp correction faction (RCF), show measurement locations.
    Outputs: 2D maps of viscoelastic parameters, csv files with output data from model,
    individual plots of contact point determination and model fitting."""

    # Set global variables to use in other functions
    global path_filename, path, filename, filename_noext, extension, csv_filename_noext

    # Define parameters required for analysis
    model = clicked.get()
    R = (float(e1.get())) / 2
    correct_drift = var1.get()
    apply_rcf = var2.get()
    show_locations = var3.get()
    path_filenames = filedialog.askopenfilenames(title="Select all files to process", filetypes = [("CSV files", "*.csv")])
    v_sample = float(e2.get())

    if path_filenames == "":
        return

    txt.insert(END, f"Converting *.txt files to *.csv files and analysing indentation depth...\n")
    txt.update()
    txt.see("end")
    txt.insert(END, "Number of files to process: {}\n\n\n".format(len(path_filenames)))
    txt.update()
    txt.see("end")

    if model == "GMM - Fixed Decay Times":
        Spacer()
        Spacer()
        txt.insert(END, "INITIATING GENERATION OF 3D MAPS WITH GENERALISED MAXWELL MODEL...\n")
        txt.update()
        txt.see("end")
        Spacer()

        # For loop to create a single set of results folders without a warning about duplicates
        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)
            Create_Output_Folder(path + f"/Indentation_Depth_Analysis")
            Create_Output_Folder(path + f"/GMM Fitting Results")
            Create_Output_Folder(path + f"/Summary and Mapping")
            break

        # Generate lists, labels and dataframes required for reporting data
        Labels_list = ['File name', 'Measurement Time [s]', 'X [um]', 'Y [um]', 'Equilibrium Modulus [kPa]', 'StDev [kPa]', 'Instantaneous Modulus [kPa]', 'StDev [kPa]', 'Viscoelastic Ratio', 'StDev'] #List of column headings for results csv file
        Results_list = [] #List of determined parameters for results csv file
        X_list = []
        Y_list = []
        Eeq_list = []
        Einst_list = []
        VR_list = []
        CR_list = []

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)
            txt.insert(END, "Analysing file {} with Generalised Maxwell Model...\n".format(filename))
            Spacer()
            try:
                path, filename_noext, Eeq, Eeq_err, Einst, Einst_err, Viscoelastic_Ratio, Viscoelastic_Ratio_err, X, Y, Contact_Radius_um, Measurement_Time = Fit_GenMaxwellModel_FixedT(path_filename, R, apply_rcf, correct_drift, v_sample)
            except ValueError:
                try:
                    df = pd.read_table(path_filename, encoding="iso-8859-1", low_memory=False,
                                           delim_whitespace=True, names=(
                        "Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]",
                        "Pos Z [um]", "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]",
                        "Gripper [V]", "Voltage A [V]", "Voltage B [V]", "Temperature [oC]"))
                    df = df[~df["Index [#]"].isin(['//'])]  # Drop the rows that contain the comments to keep only the numbers    , encoding = "iso-8859-1"
                    df = df.dropna(how='all')  # to drop if all values in the row are nan
                    df = df.astype(float)  # Change data from object to float
                # Convert previously processed csv file to df
                except:
                    df = pd.read_csv(path_filename)
                    pass

                # Get XY locations for array measurements
                PosX_um_col = df.columns.get_loc("Pos X [um]")  # Get column number
                PosY_um_col = df.columns.get_loc("Pos Y [um]")  # Get column number
                try:
                    X = np.float64(df["Pos X [um]"].median())
                    Y = np.float64(df["Pos Y [um]"].median())
                except IndexError:
                    X = df.iloc[0, PosX_um_col]
                    Y = df.iloc[0, PosY_um_col]

                filename_noext, Eeq, Eeq_err, Einst, Einst_err, Viscoelastic_Ratio, Viscoelastic_Ratio_err, Contact_Radius_um, Measurement_Time = filename, 1, 1, 1, 1, 1, 1, 1, 1

            Results_list.append(
                [filename_noext, Measurement_Time, X, Y, Eeq / 1000, Eeq_err / 1000, Einst / 1000, Einst_err / 1000,
                 Viscoelastic_Ratio, Viscoelastic_Ratio_err])

            X_list.append(X)
            Y_list.append(Y)
            Eeq_list.append(Eeq)
            Einst_list.append(Einst)
            VR_list.append(Viscoelastic_Ratio)
            CR_list.append(Contact_Radius_um)

        with open("{}/Summary and Mapping/Fitting Results Summary_GMM.csv".format(path), 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(Labels_list)
            write.writerows(Results_list)

        Spacer()
        txt.insert(END, "Summary saved in: {}/Summary and Mapping/Fitting Results Summary_GMM.csv\n".format(path))

        txt.insert(END, "X values: {}\n".format(X_list))
        txt.insert(END, "Y values: {}\n".format(Y_list))
        txt.insert(END, "Eeq values: {}\n".format(Eeq_list))

        x = X_list-min(X_list)
        y = Y_list-min(Y_list)

        fig,ax= plt.subplots(1,1)
        Eeq_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Eeq_list, 1000)), antialiased = True)
        fig.colorbar(Eeq_CP)
        ax.tricontour((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Eeq_list, 1000)), colors = 'k', antialiased = True)
        if show_locations == 1:
            circles = [plt.Circle((xi,yi), radius=ri, linewidth = 0) for xi,yi,ri in zip(x,y,CR_list)]
            c = matplotlib.collections.PatchCollection(circles)
            c.set_alpha(0.25)
            c.set_fc('grey')
            ax.add_collection(c)
        else:
            pass
        ax.set_title('Equilibrium Modulus (kPa)')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Summary and Mapping/Equilibrium Modulus Map.pdf".format(path))

        fig,ax= plt.subplots(1,1)
        Eeq_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Einst_list, 1000)), antialiased = True)
        fig.colorbar(Eeq_CP)
        ax.tricontour((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Einst_list, 1000)), colors = 'k', antialiased = True)
        if show_locations == 1:
            circles = [plt.Circle((xi,yi), radius=ri, linewidth = 0) for xi,yi,ri in zip(x,y,CR_list)]
            c = matplotlib.collections.PatchCollection(circles)
            c.set_alpha(0.25)
            c.set_fc('grey')
            ax.add_collection(c)
        else:
            pass
        ax.set_title('Instantaneous Modulus (kPa)')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Summary and Mapping/Instantaneous Modulus Map.pdf".format(path))  #, dpi=300

        fig,ax= plt.subplots(1,1)
        Eeq_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), VR_list, antialiased = True)
        fig.colorbar(Eeq_CP)
        ax.tricontour((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(VR_list, 1000)), colors = 'k', antialiased = True)
        if show_locations == 1:
            circles = [plt.Circle((xi,yi), radius=ri, linewidth = 0) for xi,yi,ri in zip(x,y,CR_list)]
            c = matplotlib.collections.PatchCollection(circles)
            c.set_alpha(0.25)
            c.set_fc('grey')
            ax.add_collection(c)
        else:
            pass
        ax.set_title('Viscoelastic Ratio')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Summary and Mapping/Viscoelastic Ratio Map.pdf".format(path))

        Spacer()
        Spacer()
        txt.insert(END, "Mapping of viscoelastic parameters complete.\n".format(year_now))
        Spacer()
        Spacer()

    if model == "GMM":
        Spacer()
        Spacer()
        txt.insert(END, "INITIATING GENERATION OF 3D MAPS WITH GENERALISED MAXWELL MODEL...\n")
        txt.update()
        txt.see("end")
        Spacer()

        # For loop to create a single set of results folders without a warning about duplicates
        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)
            csv_filename_noext, extension = os.path.splitext(filename)
            Create_Output_Folder(path + f"/Indentation_Depth_Analysis")
            Create_Output_Folder(path + f"/GMM Fitting Results")
            Create_Output_Folder(path + f"/Summary and Mapping")
            break

        # Generate lists, labels and dataframes required for reporting data
        Labels_list = ['File name', 'Measurement Time [s]', 'X [um]', 'Y [um]', 'Equilibrium Modulus [kPa]', 'StDev [kPa]', 'Instantaneous Modulus [kPa]', 'StDev [kPa]', 'Viscoelastic Ratio', 'StDev', 'Storage Modulus 1Hz [kPa]', 'Loss Modulus 1Hz [kPa]', 'Tan Delta 1Hz', 'Storage Modulus 10Hz [kPa]', 'Loss Modulus 10Hz [kPa]', 'Tan Delta 10Hz', 't1 [s]', 't2 [s]', 't3 [s]', 'PosZ_CP [um]', 'PiezoZ_CP [um]', 'E1 [kPa]', 'E2 [kPa]', 'E3 [kPa]', 'E1 Relative', 'E2 Relative', 'E3 Relative', 'Indentation depth [um]' ] #List of column headings for results csv file
        Results_list = [] # Combined list of determined parameters for results csv file
        # Individual lists for 2D mapping:
        X_list = []
        Y_list = []
        Eeq_list = []
        Einst_list = []
        VR_list = []
        CR_list = []
        Ei_1hz_list = []
        Eii_1hz_list = []
        Ei_10hz_list = []
        Eii_10hz_list = []
        TanDelta_1hz_list = []
        TanDelta_10hz_list = []

        omega = np.geomspace(0.01, 500, 100)
        omega_list = list(omega)
        omega_list.insert(0, 'Filename')
        StorageModulus_df = pd.DataFrame(columns = omega_list)
        LossModulus_df = pd.DataFrame(columns = omega_list)
        TanDelta_df = pd.DataFrame(columns = omega_list)

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)
            txt.insert(END, "Analysing file {} with Generalised Maxwell Model...\n".format(filename))
            Spacer()
            try:
                df = pd.read_table(path_filename, encoding="iso-8859-1", on_bad_lines='skip', low_memory=False,
                                   delim_whitespace=True, names=(
                        "Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]",
                        "Pos Z [um]",
                        "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]", "Gripper [V]",
                        "Voltage A [V]",
                        "Voltage B [V]", "Temperature [oC]", "Indent Depth [um]"))
                df = df[~df["Index [#]"].isin(
                    [
                        '//'])]  # Drop the rows that contain the comments to keep only the numbers    , encoding = "iso-8859-1"
                df = df.dropna(how='all')  # to drop if all values in the row are nan
                df = df.astype(float)  # Change data from object to float
            # Convert previously processed csv file to df
            except:
                df = pd.read_csv(path_filename)
                pass
            Displacement_Min_um, F_max_index, F_Min_uN, Max_Indentation_Depth_um, Ramp_Time, Force = DetermineIndentationDepth_F_relax(df, path, filename, correct_drift)

            try:
                path, filename_noext, Eeq, Eeq_err, Einst, Einst_err, Viscoelastic_Ratio, Viscoelastic_Ratio_err, X, Y, Contact_Radius_um, Measurement_Time, Ei, Eii, TanDelta, Ei_1hz, Eii_1hz, Ei_10hz, Eii_10hz, TanDelta_1hz, TanDelta_10hz, t1, t2, t3, PosZ_CP, PiezoZ_CP, E1, E2, E3 = Fit_GenMaxwellModel(path_filename, R, v_sample, correct_drift, apply_rcf)


            except (ValueError, np.linalg.LinAlgError, RuntimeError, StopIteration, IndexError): #ValueError, np.linalg.LinAlgError, RuntimeError, StopIteration, IndexError
                try:
                    df = pd.read_table(path_filename, encoding="iso-8859-1", low_memory=False,
                                           delim_whitespace=True, names=(
                        "Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]",
                        "Pos Z [um]", "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]",
                        "Gripper [V]", "Voltage A [V]", "Voltage B [V]", "Temperature [oC]"))
                    df = df[~df["Index [#]"].isin([
                                                      '//'])]  # Drop the rows that contain the comments to keep only the numbers    , encoding = "iso-8859-1"
                    df = df.dropna(how='all')  # to drop if all values in the row are nan
                    df = df.astype(float)  # Change data from object to float
                # Convert previously processed csv file to df
                except:
                    df = pd.read_csv(path_filename)
                    pass

                # Get XY locations for array measurements
                PosX_um_col = df.columns.get_loc("Pos X [um]")  # Get column number
                PosY_um_col = df.columns.get_loc("Pos Y [um]")  # Get column number
                try:
                    X = np.float64(df["Pos X [um]"].median())
                    Y = np.float64(df["Pos Y [um]"].median())
                except IndexError:
                    X = df.iloc[0, PosX_um_col]
                    Y = df.iloc[0, PosY_um_col]

                filename_noext, Eeq, Eeq_err, Einst, Einst_err, Viscoelastic_Ratio, Viscoelastic_Ratio_err, Contact_Radius_um, Measurement_Time, Ei_1hz, Eii_1hz, Ei_10hz, Eii_10hz, TanDelta_1hz, TanDelta_10hz, t1, t2, t3, PosZ_CP, PiezoZ_CP, E1, E2, E3, Ei, Eii, TanDelta = filename, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, np.empty(len(omega), dtype=object), np.empty(len(omega), dtype=object), np.empty(len(omega), dtype=object)

            # Function without try/except loop for troubleshooting:
            # path, filename_noext, Eeq, Eeq_err, Einst, Einst_err, Viscoelastic_Ratio, Viscoelastic_Ratio_err, X, Y, Contact_Radius_um, Measurement_Time, Ei, Eii, TanDelta, Ei_1hz, Eii_1hz, Ei_10hz, Eii_10hz, TanDelta_1hz, TanDelta_10hz, t1, t2, t3, PosZ_CP, PiezoZ_CP = Fit_GenMaxwellModel(path_filename, R, v_sample, correct_drift, apply_rcf)

            Results_list.append(
                [filename_noext, Measurement_Time, X, Y, Eeq / 1000, Eeq_err / 1000, Einst / 1000, Einst_err / 1000,
                 Viscoelastic_Ratio, Viscoelastic_Ratio_err, Ei_1hz / 1000, Eii_1hz / 1000, TanDelta_1hz, Ei_10hz / 1000, Eii_10hz / 1000, TanDelta_10hz, t1, t2, t3, PosZ_CP, PiezoZ_CP, E1 / 1000, E2 / 1000, E3 / 1000, E1 / Einst, E2 / Einst,  E3 / Einst, Max_Indentation_Depth_um])

            X_list.append(X)
            Y_list.append(Y)
            Eeq_list.append(Eeq)
            Einst_list.append(Einst)
            VR_list.append(Viscoelastic_Ratio)
            CR_list.append(Contact_Radius_um)
            Ei_1hz_list.append(Ei_1hz)
            Eii_1hz_list.append(Eii_1hz)
            TanDelta_1hz_list.append(TanDelta_1hz)
            Ei_10hz_list.append(Ei_10hz)
            Eii_10hz_list.append(Eii_10hz)
            TanDelta_10hz_list.append(TanDelta_10hz)

            # Dynamic Modulus Lists for frequency response plot
            Ei = list(Ei)
            Eii = list(Eii)
            TanDelta = list(TanDelta)

            Ei.insert(0, filename_noext)
            Eii.insert(0, filename_noext)
            TanDelta.insert(0, filename_noext)

            # StorageModulus_df = StorageModulus_df.append(pd.Series(Ei, index = omega_list), ignore_index=True)
            # LossModulus_df = LossModulus_df.append(pd.Series(Eii, index = omega_list), ignore_index=True)
            # TanDelta_df = TanDelta_df.append(pd.Series(TanDelta, index = omega_list), ignore_index=True)
            # txt.insert(END, "LM df: {}\n".format(LossModulus_df))

            StorageModulus_df.loc[len(StorageModulus_df)] = pd.Series(Ei, index=omega_list)
            LossModulus_df.loc[len(LossModulus_df)] = pd.Series(Eii, index=omega_list)
            TanDelta_df.loc[len(TanDelta_df)] = pd.Series(TanDelta, index=omega_list)
            # txt.insert(END, "SM df: {}\n".format(StorageModulus_df))

        # Summarise (mean and st dev for each frequency) dynamic moduli data
        Ei_df_nofilenames = StorageModulus_df.iloc[: , 1:]
        SM_means = Ei_df_nofilenames.mean(axis=0)
        SM_stdevs = Ei_df_nofilenames.std(axis=0)

        Eii_df_nofilenames = LossModulus_df.iloc[: , 1:]
        LM_means = Eii_df_nofilenames.mean(axis=0)
        LM_stdevs = Eii_df_nofilenames.std(axis=0)

        TD_df_nofilenames = TanDelta_df.iloc[: , 1:]
        TD_means = TD_df_nofilenames.mean(axis=0)
        TD_stdevs = TD_df_nofilenames.std(axis=0)

        # Define variables for plotting frequency response
        x = omega
        y1, y2, y3 = (SM_means, SM_stdevs), (LM_means, LM_stdevs), (TD_means, TD_stdevs)

        # Plot dynamic moduli summary
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Frequency (Hz)', fontsize=18)
        ax1.set_ylabel('Modulus (Pa)', fontsize=18)
        lns1 = ax1.plot(x, y1[0], color='black', label='Storage Modulus')
        ax1.fill_between(x, y1[0]-y1[1], y1[0]+y1[1], facecolor='black', alpha=0.3)
        lns2 = ax1.plot(x, y2[0], color='blue', label='Loss Modulus')
        ax1.fill_between(x, y2[0]-y2[1], y2[0]+y2[1], facecolor='blue', alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_xscale('log')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss Tangent', fontsize=18, color='red')
        lns3 = ax2.plot(x, y3[0], color='red', label='Loss Tangent')
        ax2.fill_between(x, y3[0]-y3[1], y3[0]+y3[1], facecolor='red', alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='red')

        lns = lns1 + lns2 + lns3  # To handle combined figure legend
        labs = [l.get_label() for l in lns]  # To handle combined figure legend
        ax1.legend(lns, labs, loc=0, fontsize=12)  # To handle combined figure legend
        plt.yticks(fontsize=12)
        plt.title("{}".format(f"{filename_noext}_Dynamic Moduli"), fontsize=20)
        fig.tight_layout()
        plt.savefig("{}/Summary and Mapping/Dynamic Moduli Summary.pdf".format(path), bbox_inches='tight', dpi=300)
        plt.close('all')  # To close all figures and save memory - max # of figures before warning: 20

        # Save a summary of moduli frequency response as a csv file
        with open("{}/Summary and Mapping/Dynamic Moduli Summary_GMM.csv".format(path), 'w', newline='') as f: #newline='' is necessary otherwise it creates an empty line between each record.
            write = csv.writer(f)
            Dynamic_moduli_labels = ['Frequency', 'Mean Storage Modulus [Pa]', 'Storage Modulus stdev [Pa]', 'Mean Loss Modulus [Pa]', 'Loss Modulus stdev [Pa]','Tan Delta', 'Tan Delta Error']
            DM_list = [x, y1[0], y1[1], y2[0], y2[1], y3[0], y3[1]]
            write.writerow(Dynamic_moduli_labels)
            write.writerows(zip(*DM_list))
        Spacer()
        txt.insert(END, "Summary saved in: {}/Summary and Mapping/Dynamic Moduli Summary_GMM.csv\n".format(path))

        # Save a summary of the fitting results as a csv file
        with open("{}/Summary and Mapping/Fitting Results Summary_GMM_{}.csv".format(path, csv_filename_noext), 'w', newline='') as f: #newline='' is necessary otherwise it creates an empty line between each record.
            write = csv.writer(f)
            write.writerow(Labels_list)
            write.writerows(Results_list)
        Spacer()
        txt.insert(END, "Summary saved in: {}/Summary and Mapping/Fitting Results Summary_GMM.csv\n".format(path))

        # 2D MAPPING OF VISCOELASTIC PARAMETERS
        x = X_list-min(X_list)
        y = Y_list-min(Y_list)

        fig,ax= plt.subplots(1,1)
        Eeq_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Eeq_list, 1000)), antialiased = True)
        fig.colorbar(Eeq_CP)
        ax.tricontour((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Eeq_list, 1000)), colors = 'k', antialiased = True)
        if show_locations == 1:
            circles = [plt.Circle((xi,yi), radius=ri, linewidth = 0) for xi,yi,ri in zip(x,y,CR_list)]
            c = matplotlib.collections.PatchCollection(circles)
            c.set_alpha(0.25)
            c.set_fc('grey')
            ax.add_collection(c)
        else:
            pass
        ax.set_title('Equilibrium Modulus (kPa)')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Summary and Mapping/Equilibrium Modulus Map.pdf".format(path))

        fig,ax= plt.subplots(1,1)
        Eeq_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Einst_list, 1000)), antialiased = True)
        fig.colorbar(Eeq_CP)
        ax.tricontour((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Einst_list, 1000)), colors = 'k', antialiased = True)
        if show_locations == 1:
            circles = [plt.Circle((xi,yi), radius=ri, linewidth = 0) for xi,yi,ri in zip(x,y,CR_list)]
            c = matplotlib.collections.PatchCollection(circles)
            c.set_alpha(0.25)
            c.set_fc('grey')
            ax.add_collection(c)
        else:
            pass
        ax.set_title('Instantaneous Modulus (kPa)')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Summary and Mapping/Instantaneous Modulus Map.pdf".format(path))  #, dpi=300

        fig,ax= plt.subplots(1,1)
        Eeq_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), VR_list, antialiased = True)
        fig.colorbar(Eeq_CP)
        ax.tricontour((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(VR_list, 1000)), colors = 'k', antialiased = True)
        if show_locations == 1:
            circles = [plt.Circle((xi,yi), radius=ri, linewidth = 0) for xi,yi,ri in zip(x,y,CR_list)]
            c = matplotlib.collections.PatchCollection(circles)
            c.set_alpha(0.25)
            c.set_fc('grey')
            ax.add_collection(c)
        else:
            pass
        ax.set_title('Viscoelastic Ratio')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Summary and Mapping/Viscoelastic Ratio Map.pdf".format(path))

        fig,ax= plt.subplots(1,1)
        Ei1hz_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Ei_1hz_list, 1000)), antialiased = True)
        fig.colorbar(Ei1hz_CP)
        ax.tricontour((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Ei_1hz_list, 1000)), colors = 'k', antialiased = True)
        if show_locations == 1:
            circles = [plt.Circle((xi,yi), radius=ri, linewidth = 0) for xi,yi,ri in zip(x,y,CR_list)]
            c = matplotlib.collections.PatchCollection(circles)
            c.set_alpha(0.25)
            c.set_fc('grey')
            ax.add_collection(c)
        else:
            pass
        ax.set_title('Storage Modulus 1Hz (kPa)')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Summary and Mapping/Storage Modulus 1Hz Map.pdf".format(path))

        fig,ax= plt.subplots(1,1)
        Eii1hz_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Eii_1hz_list, 1000)), antialiased = True)
        fig.colorbar(Eii1hz_CP)
        ax.tricontour((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Eii_1hz_list, 1000)), colors = 'k', antialiased = True)
        if show_locations == 1:
            circles = [plt.Circle((xi,yi), radius=ri, linewidth = 0) for xi,yi,ri in zip(x,y,CR_list)]
            c = matplotlib.collections.PatchCollection(circles)
            c.set_alpha(0.25)
            c.set_fc('grey')
            ax.add_collection(c)
        else:
            pass
        ax.set_title('Loss Modulus 1Hz (kPa)')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Summary and Mapping/Loss Modulus 1Hz Map.pdf".format(path))

        fig,ax= plt.subplots(1,1)
        Ei10hz_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Ei_10hz_list, 1000)), antialiased = True)
        fig.colorbar(Ei10hz_CP)
        ax.tricontour((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Ei_10hz_list, 1000)), colors = 'k', antialiased = True)
        if show_locations == 1:
            circles = [plt.Circle((xi,yi), radius=ri, linewidth = 0) for xi,yi,ri in zip(x,y,CR_list)]
            c = matplotlib.collections.PatchCollection(circles)
            c.set_alpha(0.25)
            c.set_fc('grey')
            ax.add_collection(c)
        else:
            pass
        ax.set_title('Storage Modulus 10Hz (kPa)')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Summary and Mapping/Storage Modulus 10Hz Map.pdf".format(path))

        fig,ax= plt.subplots(1,1)
        Eii10hz_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Eii_10hz_list, 1000)), antialiased = True)
        fig.colorbar(Eii10hz_CP)
        ax.tricontour((X_list-min(X_list)), (Y_list-min(Y_list)), (np.divide(Eii_10hz_list, 1000)), colors = 'k', antialiased = True)
        if show_locations == 1:
            circles = [plt.Circle((xi,yi), radius=ri, linewidth = 0) for xi,yi,ri in zip(x,y,CR_list)]
            c = matplotlib.collections.PatchCollection(circles)
            c.set_alpha(0.25)
            c.set_fc('grey')
            ax.add_collection(c)
        else:
            pass
        ax.set_title('Loss Modulus 10Hz (kPa)')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Summary and Mapping/Loss Modulus 10Hz Map.pdf".format(path))

        fig,ax= plt.subplots(1,1)
        TD1hz_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), (TanDelta_1hz_list), antialiased = True)
        fig.colorbar(TD1hz_CP)
        ax.tricontour((X_list-min(X_list)), (Y_list-min(Y_list)), (TanDelta_1hz_list), colors = 'k', antialiased = True)
        if show_locations == 1:
            circles = [plt.Circle((xi,yi), radius=ri, linewidth = 0) for xi,yi,ri in zip(x,y,CR_list)]
            c = matplotlib.collections.PatchCollection(circles)
            c.set_alpha(0.25)
            c.set_fc('grey')
            ax.add_collection(c)
        else:
            pass
        ax.set_title('Tan Delta 1Hz')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Summary and Mapping/Tan Delta 1Hz Map.pdf".format(path))

        fig,ax= plt.subplots(1,1)
        TD10hz_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), (TanDelta_10hz_list), antialiased = True)
        fig.colorbar(TD10hz_CP)
        ax.tricontour((X_list-min(X_list)), (Y_list-min(Y_list)), (TanDelta_10hz_list), colors = 'k', antialiased = True)
        if show_locations == 1:
            circles = [plt.Circle((xi,yi), radius=ri, linewidth = 0) for xi,yi,ri in zip(x,y,CR_list)]
            c = matplotlib.collections.PatchCollection(circles)
            c.set_alpha(0.25)
            c.set_fc('grey')
            ax.add_collection(c)
        else:
            pass
        ax.set_title('Tan Delta 10Hz')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Summary and Mapping/Tan Delta 10Hz Map.pdf".format(path))

        Spacer()
        Spacer()
        txt.insert(END, "Mapping of viscoelastic parameters complete.\n".format(year_now))
        Spacer()
        Spacer()

    if model == "Poroelastic":
        Spacer()
        Spacer()
        txt.insert(END, "INITIATING GENERATION OF 3D MAPS WITH POROELASTIC MODEL...\n")
        txt.update()
        txt.see("end")
        Spacer()

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)
            Create_Output_Folder(path + f"/Indentation_Depth_Analysis")
            Create_Output_Folder(path + f"/Poroelastic Fitting Results")
            Create_Output_Folder(path + f"/Summary and Mapping")
            break


        Labels_list = ['File name', 'Measurement Time [s]', 'X [um]', 'Y [um]', 'Equilibrium Force [uN]', 'StDev [uN]', 'Instantaneous Force [uN]', 'StDev [uN]', 'Effective Diffusivity [m2s-1]', 'StDev [m2s-1]'] #List of column headings for results csv file
        Results_list = [] # Combined list of determined parameters for results csv file
        # Individual lists for 2D mapping
        X_list = []
        Y_list = []
        Pinf_list = []
        P0_list = []
        D_list = []

        for path_filename in path_filenames:
            path, filename = os.path.split(path_filename)
            txt.insert(END, "Analysing file {} with Poroelastic Model...\n".format(filename))
            Spacer()
            path, filename_noext, Pinf_fit, Pinf_err, P0_fit, P0_err, D_fit, D_err, X, Y, Contact_Radius_um, Measurement_Time, PosZ_CP, PiezoZ_CP = Fit_PoroelasticModel(path_filename, R, correct_drift)
            Results_list.append([filename_noext, Measurement_Time, X, Y, Pinf_fit*1000000, Pinf_err*1000000, P0_fit*1000000, P0_err*1000000, D_fit, D_err])
            X_list.append(X)
            Y_list.append(Y)
            Pinf_list.append(Pinf_fit)
            P0_list.append(P0_fit)
            D_list.append(D_fit)


        with open("{}/Fitting Results Summary_Poroelastic.csv".format(path + f"/Summary and Mapping"), 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(Labels_list)
            write.writerows(Results_list)

        Spacer()
        txt.insert(END, "Summary saved in: {}/Fitting Results Summary_Poroelastic.csv\n".format(path + f"/Summary and Mapping"))

        #txt.insert(END, "X values: {}\n".format(X_list))
        #txt.insert(END, "Y values: {}\n".format(Y_list))
        #txt.insert(END, "Eeq values: {}\n".format(Eeq_list))


        fig,ax= plt.subplots(1,1)
        Eeq_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), (np.multiply(Pinf_list, 1000000)), antialiased = True)
        fig.colorbar(Eeq_CP)
        ax.set_title('Equilibrium Force (uN)')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Equilibrium Force Map.pdf".format(path + f"/Summary and Mapping"))

        fig,ax= plt.subplots(1,1)
        Eeq_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), (np.multiply(P0_list, 1000000)), antialiased = True)
        fig.colorbar(Eeq_CP)
        ax.set_title('Instantaneous Modulus (uN)')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Instantaneous Force Map.pdf".format(path + f"/Summary and Mapping"))  #, dpi=300

        fig,ax= plt.subplots(1,1)
        Eeq_CP = ax.tricontourf((X_list-min(X_list)), (Y_list-min(Y_list)), D_list, antialiased = True)
        fig.colorbar(Eeq_CP)
        ax.set_title('Effective Diffusivity [m2s-1]')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.savefig("{}/Effective Diffusivity Map.pdf".format(path + f"/Summary and Mapping"))

        Spacer()
        Spacer()
        txt.insert(END, "Mapping of poroelastic parameters complete.\n".format(year_now))
        Spacer()
        Spacer()

    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")


def Divide_Array_Data_b4():
    """Button to divide txt file(s) containing an array of indentation measurements
     into individual csv files for each indentation."""
    # Set global variables
    global path_filename, path, filename, filename_noext, extension, folder_path_Array_Measurements

    # OPEN FILE
    path_filenames = filedialog.askopenfilenames(initialdir = "/Users/pierangelogobbo/Dropbox/@Documents/@My Research/Lab useful docs/FemtoTools Nanoindenter/Indentation_Curve_Analysis/Input files", title = "Select a file", filetypes = (("Text files", "*.txt"),("All files","*.*")))
    if path_filenames == "":
        return

    # Create a single output folder to store all individual measurements within
    for path_filename in path_filenames:
        path, filename = os.path.split(path_filename)
        filename_noext, extension = os.path.splitext(filename)
        folder_path_Array_Measurements = path + f"/{filename_noext}_Measurements"
        Create_Output_Folder(folder_path_Array_Measurements)
        break

    Spacer()
    txt.insert(END, "Dividing array files into individual *.txt files for each measurement. Number of array files to process: {}\n".format(len(path_filenames)))
    txt.update()
    txt.see("end")
    Spacer()

    for path_filename in path_filenames:
        path, filename = os.path.split(path_filename)
        Spacer()
        txt.insert(END, "Dividing array file {} into individual *.txt files for each measurement...\n".format(filename))
        txt.update()
        txt.see("end")

        Divide_Array_Data(path_filename, folder_path_Array_Measurements)

    Spacer()
    Spacer()
    txt.insert(END, "Individual measurement files saved in directory: {}\n".format(folder_path_Array_Measurements))
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")

def ArrayCoord():

    global path_filename, path, filename, filename_noext, extension

    path_filenames = filedialog.askopenfilenames(title="Select all files to process", filetypes=[("CSV files", "*.csv")])

    if path_filenames == "":
        return

    for path_filename in path_filenames:
        path, filename = os.path.split(path_filename)
        txt.insert(END, "Analysing file {}...\n".format(filename))
        Spacer()
        try:
            df = pd.read_table(path_filename, encoding="iso-8859-1", low_memory=False,
                                   delim_whitespace=True, names=("X [um], Y [um]"))
            df = df.dropna(how='all')  # to drop if all values in the row are nan
            df = df.astype(float)
        except:
            df = pd.read_csv(path_filename)
            pass

        #Get columns from df
        X_col = df.columns.get_loc("X [um]")
        Y_col = df.columns.get_loc("Y [um]")

        #Define array of data
        X = np.array(df.iloc[:, X_col])
        Y = np.array(df.iloc[:, Y_col])

        X_list = sorted(X)
        n = len(X_list)

        with open("{}/Array Coordinates.csv".format(path), 'w', newline='') as f:
            Labels_list = ['X', 'Y']
            write = csv.writer(f)
            write.writerow(Labels_list)

            for i in range(n):
                PosX_new = round((X[i] - X[0]),1)
                PosY_new = round((Y[i] - Y[0]),1)

                Results_list = []
                Results_list.append([PosX_new, PosY_new])
                write.writerows(Results_list)

            Spacer()
            txt.insert(END, "Coordinates saved in: {}/Array Coordinates.csv\n".format(path))
            Spacer()
            Spacer()
            txt.insert(END, "END OF PROGRAM.\n".format(year_now))
            Spacer()
            Spacer()
            txt.update()
            txt.see("end")

            return PosX_new, PosY_new

def DetermineIndentationDepth_b9():
    """Button to determine indentation depth from recording of DMA measurements.
    Input ratio BL points and threshold constant."""
    # Define variables
    global path_filename
    ratio_BL_points = float(e3.get())
    threshold_constant = float(e4.get())

    # Open files
    path_filenames = filedialog.askopenfilenames(initialdir="/Users/pierangelogobbo/Dropbox/@Documents/@My Research/Lab useful docs/FemtoTools Nanoindenter/Indentation_Curve_Analysis/Input files", title="Select all files to process", filetypes = [("TXT files", "*.txt")])
    if path_filenames == "":
        return

    txt.insert(END, f"Converting *.txt files to *.csv files and analysing indentation depth...\n")
    txt.update()
    txt.see("end")
    txt.insert(END, "Number of files to process: {}\n\n\n".format(len(path_filenames)))
    txt.update()
    txt.see("end")

    for path_filename in path_filenames:
        DetermineIndentationDepth_DMA(path_filename, ratio_BL_points, threshold_constant)

    Spacer()
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")


def Gen_Single_Exp_b6():
    """Button to divide txt file containing multiple (array or single) measurements
    into individual txt files for each measurement."""
    # Set global variables
    global path_filename, path, filename, filename_noext, extension, folder_path

    # Ask to open file and get filename and path
    path_filename = filedialog.askopenfilename(initialdir = "/Users/pierangelogobbo/Dropbox/@Documents/@My Research/Lab useful docs/FemtoTools Nanoindenter/Indentation_Curve_Analysis/Input files", title = "Select a file", filetypes = (("Text files", "*.txt"),("All files","*.*")))

    path, filename, filename_noext, extension, folder_path = Find_Experiments(path_filename)

    Spacer()
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")


def DMA_analysis_b10():
    """Button to carry out conversion of DMA output data to viscoelastic moduli.
    Input probe diameter, Poisson's Ratio, and indentation depth."""
    # Set global variables
    global path_filename, path, filename, filename_noext, extension, folder_path

    R = (float(e1.get())) / 2
    v_sample = float(e2.get())
    Ind_Dep_um = float(e5.get())
    path, filename = os.path.split(path_filename)

    # Ask to open file and get filename and path
    path_filename = filedialog.askopenfilename(initialdir = "/Users/pierangelogobbo/Dropbox/@Documents/@My Research/Lab useful docs/FemtoTools Nanoindenter/Indentation_Curve_Analysis/Input files", title = "Select a file", filetypes = (("Text files", "*.txt"),("All files","*.*")))

    DMA_Analysis(path_filename, R, v_sample, Ind_Dep_um)

    Spacer()
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")

#INTERFACE
l1 = Label(root, text="A.L.I.A.S.: A Lovely Indentation Analysis System - Viscoelasticity", font='Helvetica 24 bold', fg = "SteelBlue4").grid(row = 0, column = 0, sticky = W, padx = 200, pady = 2)

l3 = Label(root, text="Experimental and fitting parameters:", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 1, column = 0, sticky = W, padx = 5, pady = 2)
l4 = Label(root, text="Probe's diameter (um):").grid(row = 2, column = 0, sticky = W, padx = 5, pady = 2)
e1 = Entry(root, width=5)
e1.insert(END, "260")
e1.grid(row = 2, column = 0, sticky = W, padx = 350, pady = 2)

l5 = Label(root, text="Sample's Poisson's ratio:").grid(row = 3, column = 0, sticky = W, padx = 5, pady = 2)
e2 = Entry(root, width=5)
e2.insert(END, "0.5") # Poisson's ratio of sample (v_PDMS 0.45-0.5; v_polystyrene 0.34)
e2.grid(row = 3, column = 0, sticky = W, padx = 350, pady = 2)

l6 = Label(root, text="Baseline data points (default: first 1/8 of datapoints):").grid(row = 4, column = 0, sticky = W, padx = 5, pady = 2)
e3 = Entry(root, width=5)
e3.insert(END, "0.125") #Number of points for baseline is 1/8 of datapoints of df
e3.grid(row = 4, column = 0, sticky = W, padx = 350, pady = 2)

l7 = Label(root, text="Contact point threshold constant (*st.dev.):").grid(row = 5, column = 0, sticky = W, padx = 5, pady = 2)
e4 = Entry(root, width=5)
e4.insert(END, "20") #Number of points for baseline is 1/8 of datapoints of df
e4.grid(row = 5, column = 0, sticky = W, padx = 350, pady = 2)

l2 = Label(root, text="Data Processing:", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 7, column = 0, sticky = W, padx = 5, pady = 2)
b1 = Button(root, text="Extract data", command=File_Extractor_b1).grid(row = 8, column = 0, sticky = W, padx = 5, pady = 2)
b6 = Button(root, text="Divide Data", command=Gen_Single_Exp_b6).grid(row = 8, column = 0, sticky = W, padx = 130, pady = 2)
b4 = Button(root, text="Divide array data", command=Divide_Array_Data_b4).grid(row = 8, column = 0, sticky = W, padx = 260, pady = 2)


l9 = Label(root, text="Force relaxation analysis:", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 9, column = 0, sticky = W, padx = 5, pady = 2)
l10 = Label(root, text="Select fitting model:").grid(row = 10, column = 0, sticky = W, padx = 5, pady = 2)
options = ["GMM", "GMM - Fixed Decay Times","Poroelastic"]
clicked = StringVar()
clicked.set(options[0])
dm1 = OptionMenu(root, clicked, *options).grid(row = 10, column = 0, sticky = W, padx = 150, pady = 2)
b2 = Button(root, text="Analyse individual measurements", command=ForceRelaxationAnalysis_Individual_b2).grid(row = 11, column = 0, sticky = W, padx = 5, pady = 2)
b3 = Button(root, text="Analyse array measurements", command=ForceRelaxationAnalysis_Array_b3).grid(row = 12, column = 0, sticky = W, padx = 5, pady = 2)
var1 = IntVar()
var2 = IntVar()
var3 = IntVar()
c1 = Checkbutton(root, text='Correct Drift', variable=var1).grid(row = 10, column = 0, sticky = W, padx = 320, pady = 0)
c2 = Checkbutton(root, text='Apply RCF', variable=var2).grid(row = 11, column = 0, sticky = W, padx = 320, pady = 0)
c3 = Checkbutton(root, text='Show Locations', variable=var3).grid(row = 12, column = 0, sticky = W, padx = 320, pady = 0)
b5= Button(root, text="Get array coordinates", command=ArrayCoord).grid(row = 13, column = 0, sticky = W, padx = 5, pady = 2)

l12 = Label(root, text="DMA analysis", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 14, column = 0, sticky = W, padx = 5, pady = 2)
l8 = Label(root, text="Indentation Depth (um):").grid(row = 15, column = 0, sticky = W, padx = 5, pady = 2)
e5 = Entry(root, width=5)
e5.insert(END, "5") #Indentation depth determined for DMA measurement
e5.grid(row = 15, column = 0, sticky = W, padx = 160, pady = 2)
b9 = Button(root, text="Determine Depth", command=DetermineIndentationDepth_b9).grid(row = 16, column = 0, sticky = W, padx = 5, pady = 2)
b10 = Button(root, text="Run DMA Analysis", command=DMA_analysis_b10).grid(row = 16, column = 0, sticky = W, padx = 160, pady = 2)


#Create and write inside a dialog box
l15 = Label(root, text="Dialog window:", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 1, column = 0, sticky = W, padx = 450, pady = 2)
txt = scrolledtext.ScrolledText(root, height=30, width=95)
txt.configure(font=("TkDefaultFont", 12, "normal"))
txt.grid(row=2, column = 0, rowspan = 17, sticky=W, padx = 450) #W=allign to left
txt.see("end")

root.mainloop()