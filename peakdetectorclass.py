import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from QRSDetectorOffline import QRSDetectorOffline
from time import gmtime, strftime
from scipy.signal import butter, lfilter, sosfiltfilt
import math as m
from scipy.signal import find_peaks, peak_prominences, peak_widths
import statistics

class peakdetectorclass(object):

    def column(self, matrix, i):
        return [row[i] for row in matrix]
    
    def __init__(self, QRSDetector, All_Data, FIlter = True, NormalSCG = False, Name = None, IntervalH = 150, IntervalL = 50):
        
        self.name = Name

        self.All_Data = All_Data
        self.qrs_detector = QRSDetector
        self.X_valuesLong = np.arange(0,len(QRSDetector.ecg_data_raw), 1) + QRSDetector.SkipRow
        self.X_values_ms = self.column(All_Data, 0)
        self.ECG_Raw  = self.column(All_Data, 1)
        self.Accelerometer_Raw_X = self.column(All_Data, 2)
        self.Accelerometer_Raw_Y = self.column(All_Data, 3)
        self.Accelerometer_Raw_Z = self.column(All_Data, 4)
        self.BCG_Raw = self.column(All_Data, 5)


        self.ECG_norm = self.normalize(self.ECG_Raw)
        self.Accelerometer_Norm_X = self.normalize(self.Accelerometer_Raw_X)
        self.Accelerometer_Norm_Y = self.normalize(self.Accelerometer_Raw_Y)
        self.Accelerometer_Norm_Z = self.normalize(self.Accelerometer_Raw_Z)
        self.BCG_norm = self.normalize(self.BCG_Raw)# *-1

        if NormalSCG:
            self.Accelerometer_Norm_Z =np.sqrt(np.array(self.Accelerometer_Norm_Z)**2+ 
                                               np.array(self.Accelerometer_Norm_Y)**2+
                                               np.array(self.Accelerometer_Norm_X)**2)



        self.PanTompkinsPoints = []

        self.ExactpointsX_ECG = []
        self.ExactpointsY_ECG = []

        self.ExactpointsX_Acc_Z = []
        self.ExactpointsY_Acc_Z = []

        self.IntervalHigh = IntervalH #150 #30 
        self.IntervalLow = IntervalL #50 #60

        self.lowcut = 1
        self.highcut = 30
        self.order = 3

        self.order_lowpass = 3

        self.lowcut_acc = 3
        self.highcut_acc = 30

        self.IntervalHighACC = 600
        self.IntervalLowACC = 0


        self.averageECG = []
        self.averageACC = []
        self.averageBCG = []
        self.averageACC_raw = []
        self.averageACCPeak = []
        self.averageECGPeak = []
        self.averageBCGPeak = []
        self.averageACCPeak_raw = []

        self.averageACC_filtered = []
        self.averageBCG_filtered= []

        self.AmplitudeValues = []
        
        self.PEP = []
        self.betterPEP = []
        self.PEPB = []
        self.SBD = [] #seismo balisto delay
        self.AccAmplitude = []
        self.RtoI = []
        self.ItoJ = []
        self.IJ_delta_height = []
        self.RMS_height = []
        self.RMS_width = []
        self.respiratoryRate = []

        self.BCG_Filtered = self.bandpass_filter(self.BCG_norm, low=self.lowcut, high= self.highcut)
        self.BCG_Filtered_lowpass = self.lowpass_filter(self.BCG_Raw, cutoff=0.5)

        self.ACC_Filtered = self.bandpass_filter(self.Accelerometer_Norm_Z,low= self.lowcut_acc, high= self.highcut_acc)
        
        
        
        self.addPoints()
        self.addExactPoints()
        self.averageSignal()
        self.amplitude()


        length_seconds = self.X_values_ms[-1]/60000

        self.pepXaxis = np.linspace(start=0, stop=length_seconds, num = len(self.betterPEP))

       

        if FIlter:
            self.Accelerometer_Norm_Z = self.ACC_Filtered
            self.BCG_norm = self.BCG_Filtered
        
        

 




    def addPoints(self):
        for p in self.qrs_detector.qrs_peaks_indices:
                self.PanTompkinsPoints.append([p,self.qrs_detector.ecg_data_detected[p][1]])
    
    def isolatedBeat(self, instance):
        start = self.qrs_detector.qrs_peaks_indices[instance]
        self.low = start-self.IntervalLow
        if self.low < 0 : self.low = 0 
        self.high = start+self.IntervalHigh
        self.X_islolated_val = np.arange(self.low, self.high, 1)
        self.isolatedECG = self.ECG_norm[self.low: self.high: 1]


    def islolatedACC(self, instance):
        #self.startAcc = self.ExactpointsX_ECG[instance] - self.X_valuesLong[0]
        #self.highACC = self.startAcc + self.IntervalHighACC  
        #self.lowACC = self.startAcc + self.IntervalLowACC
        start = self.qrs_detector.qrs_peaks_indices[instance]
        self.lowACC = start-self.IntervalLow
        if self.lowACC < 0 : self.lowACC = 0 
        self.highACC = start+self.IntervalHigh

        self.IsolatedAccelermeterX = self.Accelerometer_Norm_X[self.lowACC: self.highACC: 1]
        self.IsolatedAccelermeterY= self.Accelerometer_Norm_Y[self.lowACC: self.highACC: 1]
        self.IsolatedAccelermeterZ = self.Accelerometer_Norm_Z[self.lowACC: self.highACC: 1]

        self.Isolated_Z_NonNormalized = self.Accelerometer_Raw_Z[self.lowACC: self.highACC: 1]
        self.Isolated_Z_filtered = self.ACC_Filtered[self.lowACC: self.highACC: 1]

        self.X_islolated_val_acc = np.arange(self.lowACC, self.highACC, 1)

        #self.X_islolated_val_derived = np.arange(self.startAcc, self.highACC -1, 1)
    

    """         X_poly = np.arange(self.startAcc, self.highACC, 1)
        self.polyfitACC_Z = self.Polyfit(self.X_islolated_val_acc, self.IsolatedAccelermeterZ)
        self.DerivedIsolatedZ = self.deriveSegment(self.polyfitACC_Z(X_poly)) """
        
 
    def isolateBCG(self, instance):
        start = self.qrs_detector.qrs_peaks_indices[instance]
        self.lowBCG = start-self.IntervalLow
        if self.lowBCG < 0 : self.lowBCG = 0 
        self.highBCG = start+self.IntervalHigh
        self.X_islolated_val_BCG = np.arange(self.lowBCG, self.highBCG, 1)
        self.isolatedBCG = self.BCG_norm[self.lowBCG: self.highBCG: 1]
        self.isolatedBCG_filtered = self.BCG_Filtered[self.lowBCG: self.highBCG: 1]
   
 
        """     def normalize(self, data):
        mean = np.mean(data)
        meanList = np.full_like(data, mean)
        SD = statistics.pstdev(data)
        return (data-meanList)/SD """

    """ def normalize(self, data):
        min = np.min(data)
        max = np.max(data)
        return (data - min) / (max - min) """

    def normalize(self, data):
        
        return (data / np.linalg.norm(data))       
           
    
    def FindPeak2(self, IsolatedData):
        instance = -1
        max = -10
        x_value = 0
        for i in IsolatedData:
            instance +=1
            if i > max:
                max = i
                x_value = instance
        x_value = x_value + self.low + self.X_valuesLong[0]
        return [x_value, max]
    
    def FindPeak(self, IsolatedData, ECG = False, BCG = False):
        x = np.argmax(IsolatedData)
        y = np.max(IsolatedData)
        point = [x, y]

        if ECG:
            p,_ = find_peaks(IsolatedData[25:50],distance=150)#, prominence=0.00002)
            try:
                point=[p[0]+25, IsolatedData[p[0]+25]]
            except Exception as e:
                print(e)
                print('ECG peak error')
        
        if BCG:
            start= 40
            end = 85
            p,_ = find_peaks(IsolatedData[start:end] *-1 , distance= 300) #0.0002
            try:
                point=[p[0]+start, IsolatedData[p[0]+start]]
            except Exception as e:
                print(e)
                print('BCG peak error')

        return(point)
    
    def FindLowPeak(self, IsolatedData, SCG = False):
        
        start = 60
        end = 80
        ID = IsolatedData[start:end] #80
        x = np.argmin(ID)
        y = np.min(ID)
        point = [x+start, y]


        if SCG:
            d = (ID) *-1
            d = d + np.ones_like(d)
            peaks, _=find_peaks(d, distance=300)
            try:
                point = [peaks[0]+start, d[peaks[0]]]
            except Exception as e:
                print(e)
                print("LowPeak error")
        return(point)
        
    
    def FindPeakACC(self, IsolatedData):
        instance = -1
        max = 10
        x_value = 0
        for i in IsolatedData:
            instance +=1
            if i < max:
                max = i
                x_value = instance
        x_value = x_value + self.X_valuesLong[0] + self.lowACC
        return [x_value, max]
    
    def deriveSegment(self, data):
        return np.ediff1d(data)
    
    def Polyfit(self, dataX, DataY):
        Datapoly=np.polyfit(dataX,DataY,80)
        return np.poly1d(Datapoly)

    def addExactPoints(self):
        for p in range(len(self.PanTompkinsPoints)):
            point = []
            self.isolatedBeat(p)

            point = self.FindPeak2(self.isolatedECG)
            self.ExactpointsX_ECG.append(point[0])
            self.ExactpointsY_ECG.append(point[1])
            pepECG = point[0]  

            self.islolatedACC(p)
            point = self.FindPeakACC(self.IsolatedAccelermeterZ)
            self.ExactpointsX_Acc_Z.append(point[0])
            self.ExactpointsY_Acc_Z.append(point[1])
            pepACC = point[0]

            self.PEP.append(pepACC - pepECG)
        

    
    def averageSignal(self):
        counter = 0
        beats = []
        Xaxis = np.array([])
        self.PulseX = []
        self.PulseY = np.array([])
        averageAcc = np.zeros(self.IntervalHigh+self.IntervalLow)
        averageECG = np.zeros(self.IntervalHigh+self.IntervalLow)
        averageBCG = np.zeros(self.IntervalHigh+self.IntervalLow) 

        RawAcc = np.zeros(self.IntervalHigh+self.IntervalLow)


        Filtered_Acc = np.zeros(self.IntervalHigh+self.IntervalLow)
        Filtered_BCG = np.zeros(self.IntervalHigh+self.IntervalLow)


        for point in range(len(self.PanTompkinsPoints)):
            self.isolatedBeat(point)
            self.islolatedACC(point)
            self.isolateBCG(point)
            
            
            try: 

                averageAcc += self.IsolatedAccelermeterZ# / np.linalg.norm(self.IsolatedAccelermeterZ)
                averageECG += self.isolatedECG# / np.linalg.norm(self.isolatedECG)
                averageBCG += self.isolatedBCG# / np.linalg.norm(self.isolatedBCG)
                RawAcc += self.Isolated_Z_NonNormalized
                beats.append(self.qrs_detector.qrs_peaks_indices[point])
                counter += 1

            except Exception as e: 
                print('error: ')
                print(e)
                

            if counter >= 10:
                averageECG = averageECG / counter
                averageAcc = averageAcc / counter
                averageBCG = averageBCG / counter
                RawAcc = RawAcc / counter

                #Filtered_Acc / counter
                #Filtered_BCG / counter


                ecg = self.FindPeak(averageECG, ECG=True)
                acc = self.FindLowPeak(averageAcc, SCG=True)
                #bcg, property = find_peaks(averageBCG, prominence=0.001, width=(None,20))
                bcg = self.FindPeak(averageBCG, BCG=True)
                bcgI = self.FindLowPeak(averageBCG)

                accpoint = self.FindLowPeak(RawAcc, SCG =True)

                self.AccAmplitude.append(accpoint[1])

                self.betterPEP.append((acc[0] - ecg[0])/self.qrs_detector.signal_frequency *1000)
                self.PEPB.append((bcg[0] - ecg[0]) /self.qrs_detector.signal_frequency *1000)
                self.SBD.append((bcg[0] - acc[0]) /self.qrs_detector.signal_frequency *1000)
                self.RtoI.append((bcgI[0] - ecg[0]) /self.qrs_detector.signal_frequency *1000)
                self.ItoJ.append((bcg[0] - bcgI[0]) /self.qrs_detector.signal_frequency *1000)
                self.IJ_delta_height.append((bcg[1] - bcgI[1])*100)

                self.averageACC.append(averageAcc)
                self.averageECG.append(averageECG)
                self.averageBCG.append(averageBCG)
                self.averageACC_raw.append(RawAcc)

                #self.averageACC_filtered.append(Filtered_Acc)
                #self.averageBCG_filtered.append(Filtered_BCG)


                self.averageACCPeak.append(acc)
                self.averageECGPeak.append(ecg)
                self.averageBCGPeak.append(bcg)
                self.averageACCPeak_raw.append(accpoint)

                h, w = self.rollingRMS(averageAcc)
                self.RMS_height.append(h)
                self.RMS_width.append(w)



                averageAcc = np.zeros(self.IntervalHigh+self.IntervalLow)
                averageECG = np.zeros(self.IntervalHigh+self.IntervalLow) 
                averageBCG = np.zeros(self.IntervalHigh+self.IntervalLow)

                min = np.min(beats)
                max = np.max(beats)
                minutes = ((max-min)/self.qrs_detector.signal_frequency) / 60 #antall minutter brukt pp 10 slag
                self.PulseX.append((len(beats)/minutes))


                beats = []
                counter = 0
                #print(beats)
                #pulse = len(beats) / ((beats[-1] - beats[0]) / self.qrs_detector.signal_frequency)
                #a = np.arange(len(Xaxis), dtype=np.double)
                #np.append(self.PulseY, np.full_like(a, pulse))
                #np.append(self.PulseX, Xaxis)

                #np.delete(Xaxis)
                #np.delete(beats)



    
    def bandpass_filter(self, data, low = 0.5, high = 30):
    
        nyquist_freq = 0.5 * self.qrs_detector.signal_frequency
        filterlow = low / nyquist_freq
        filterhigh = high / nyquist_freq
        #b, a = butter(self.order , [filterlow, filterhigh], btype="band")
        sos = butter(self.order , [filterlow, filterhigh], btype="band", output='sos')
        #y = lfilter(b, a, data)
        y = sosfiltfilt(sos, data)
        return y
    
    def lowpass_filter(self, data, cutoff = 1):
        nyquist_freq = 0.5 * self.qrs_detector.signal_frequency
        normalCutoff = cutoff/nyquist_freq
        sos = butter(self.order_lowpass, normalCutoff, btype='low', output='sos' )
        y = sosfiltfilt(sos, data)
        return y

    def amplitude(self):

        for i in range(len(self.averageACC)):
            start = 50
            end = 100
            data = self.averageACC[i][start:end] *1000000
            self.AmplitudeValues.append(data.max() - data.min())
                
            """             try:
                
                #AmplitudeInterval = self.averageACC[i][self.averageACCPeak[i][0]-(low): self.averageACCPeak[i][0]+20]
                
                #self.AmplitudeValues.append(np.max(AmplitudeInterval)-np.min(AmplitudeInterval))
                data = self.averageACC[i]
                high = self.FindLowPeak(data)[1]
                low = self.FindLowPeak(data * -1)[1] 
                self.AmplitudeValues.append(1000*(high - low))
            except Exception as e: 
                print('error: ')
                print(e) """
            
                

    def before_after_metrics(self, SeparationInstance , ValuesBefore, ValuesAfter, tolerance = 2): #SeparationInstance in index 
        Index = np.arange(start = 1,stop =  max(ValuesAfter, ValuesBefore) +2 - tolerance, step = 1)

        RIlistB = np.zeros((len(Index),), dtype= float)
        RJlistB = np.zeros((len(Index),), dtype= float) 
        AmplitudeB = np.zeros((len(Index),), dtype= float) 
        RtoIB = np.zeros((len(Index),), dtype= float)
        ItoJB = np.zeros((len(Index),), dtype= float)
        IJ_deltaB = np.zeros((len(Index),), dtype= float)
        PulseB = np.zeros((len(Index),), dtype= float)
        RMS_heightB = np.zeros((len(Index),), dtype= float)
        RMS_widthB = np.zeros((len(Index),), dtype= float)


        RIlistA = np.zeros((len(Index),), dtype= float) 
        RJlistA = np.zeros((len(Index),), dtype= float) 
        AmplitudeA = np.zeros((len(Index),), dtype= float)
        RtoIA = np.zeros((len(Index),), dtype= float)
        ItoJA = np.zeros((len(Index),), dtype= float)
        IJ_deltaA = np.zeros((len(Index),), dtype= float)
        PulseA = np.zeros((len(Index),), dtype= float)
        RMS_heightA = np.zeros((len(Index),), dtype= float)
        RMS_widthA = np.zeros((len(Index),), dtype= float)

        
        RJ = np.array(self.replace_outliers_iqr(self.PEPB, factor=5)) #0.05
        Amp = np.array(self.replace_outliers_iqr(self.AmplitudeValues, factor = 5))
        RI = np.array( self.replace_outliers_iqr(self.betterPEP, factor=8))
        RtoI_ = self.replace_outliers_iqr(self.RtoI,factor=5)
        ItoJ_ = self.replace_outliers_iqr(self.ItoJ, factor=2)
        IJdelta = self.replace_outliers_iqr(self.IJ_delta_height, factor=10)
        Pulse = self.replace_outliers_iqr(self.PulseX, factor= 1.5)
        RMSW = self.RMS_width
        RMSH = self.RMS_height


        for i in range(ValuesBefore - tolerance):
            iterator = SeparationInstance - ValuesBefore + i
            RIlistB[i] = RI[iterator]
            RJlistB[i] = RJ[iterator]
            AmplitudeB[i] = Amp[iterator] 
            RtoIB[i] = RtoI_[iterator]
            ItoJB[i] = ItoJ_[iterator]
            IJ_deltaB[i] = IJdelta[iterator]
            PulseB[i] = Pulse[iterator]
            RMS_heightB[i] = RMSH[iterator]
            RMS_widthB[i] = RMSW[iterator]
        for i in range(ValuesAfter - tolerance):

            iterator = SeparationInstance + i + tolerance
            RIlistA[i] = RI[iterator]
            RJlistA[i] = RJ[iterator]
            AmplitudeA[i] = Amp[iterator]
            RtoIA[i] = RtoI_[iterator]
            ItoJA[i] = ItoJ_[iterator]
            IJ_deltaA[i] = IJdelta[iterator]
            PulseA[i] = Pulse[iterator]
            RMS_heightA[i] = RMSH[iterator]
            RMS_widthA[i] = RMSW[iterator]
        
        RIlistB[RIlistB == 0] = np.nan
        RIlistB[-1] = np.nanmean(RIlistB)

        RJlistB[RJlistB == 0] = np.nan
        RJlistB[-1] = np.nanmean(RJlistB)

        AmplitudeB[AmplitudeB == 0] = np.nan
        AmplitudeB[-1] = np.nanmean(AmplitudeB)

        RIlistA[RIlistA == 0] = np.nan
        RIlistA[-1] = np.nanmean(RIlistA)

        RJlistA[RJlistA == 0] = np.nan
        RJlistA[-1] = np.nanmean(RJlistA)

        AmplitudeA[AmplitudeA == 0] = np.nan
        AmplitudeA[-1] = np.nanmean(AmplitudeA)

        RtoIB[RtoIB == 0] = np.nan
        RtoIB[-1] = np.nanmean(RtoIB)

        ItoJB[ItoJB == 0] = np.nan
        ItoJB[-1] = np.nanmean(ItoJB)

        IJ_deltaB[IJ_deltaB == 0] = np.nan
        IJ_deltaB[-1] = np.nanmean(IJ_deltaB)

        RtoIA[RtoIA == 0] = np.nan
        RtoIA[-1] = np.nanmean(RtoIA)

        ItoJA[ItoJA == 0] = np.nan
        ItoJA[-1] = np.nanmean(ItoJA)

        IJ_deltaA[IJ_deltaA == 0] = np.nan
        IJ_deltaA[-1] = np.nanmean(IJ_deltaA)

        PulseB[PulseB == 0] = np.nan
        PulseB[-1] = np.nanmean(PulseB)

        PulseA[PulseA == 0] = np.nan
        PulseA[-1] = np.nanmean(PulseA)

        RMS_heightB[RMS_heightB == 0] = np.nan
        RMS_heightB[-1] = np.nanmean(RMS_heightB)

        RMS_heightA[RMS_heightA == 0] = np.nan
        RMS_heightA[-1] = np.nanmean(RMS_heightA)

        RMS_widthA[RMS_widthA == 0] = np.nan
        RMS_widthA[-1] = np.nanmean(RMS_widthA)

        RMS_widthB[RMS_widthB == 0] = np.nan
        RMS_widthB[-1] = np.nanmean(RMS_widthB)




        d = {'ECG R-AO SCG Before': RIlistB, 
             'ECG R - J BCG Before': RJlistB, 
             'SCG amplitude Before': AmplitudeB,
             'ECG R - I BCG Before': RtoIB,
             'BCG I - J BCG Before' :ItoJB,
             'BCG I - J BCG Delta height Before' :IJ_deltaB,
             'Pulse Bfore' :PulseB,
             'RMS width Before': RMS_widthB,
             'RMS height Before': RMS_heightB,
             'ECG R-AO SCG After': RIlistA, 
             'ECG R - J BCG After': RJlistA, 
             'SCG amplitude After': AmplitudeA,
             'ECG R - I BCG After': RtoIA,
             'BCG I - J BCG After' :ItoJA,
             'BCG I - J BCG Delta height After' :IJ_deltaA,
             'Pulse After' :PulseA,
             'RMS width After':RMS_widthA,
             'RMS height After':RMS_heightA}
        
        df = pd.DataFrame(data=d, index=np.append(Index[:-1],'Average'))
        df.name = self.name 
        return df
    

    
    def replace_outliers_iqr(self, y_values_input, factor=0.5):
        # Calculate Q1, Q3, and IQR
        y_values = y_values_input.copy()
        q1 = np.percentile(y_values, 25)
        q3 = np.percentile(y_values, 75)
        iqr = q3 - q1
        
        # Calculate the lower and upper bounds for outliers
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        # Replace outliers with estimated values using linear interpolation
        interpolated_values = np.copy(y_values)
        for i in range(len(interpolated_values)):
            if interpolated_values[i] < lower_bound or interpolated_values[i] > upper_bound:
                if i == 0:
                    interpolated_values[i] = np.interp(i, [j for j in range(len(interpolated_values)) if interpolated_values[j] >= lower_bound and interpolated_values[j] <= upper_bound], [interpolated_values[j] for j in range(len(interpolated_values)) if interpolated_values[j] >= lower_bound and interpolated_values[j] <= upper_bound])
                elif i == len(interpolated_values)-1:
                    interpolated_values[i] = np.interp(i, [j for j in range(len(interpolated_values)) if interpolated_values[j] >= lower_bound and interpolated_values[j] <= upper_bound], [interpolated_values[j] for j in range(len(interpolated_values)) if interpolated_values[j] >= lower_bound and interpolated_values[j] <= upper_bound])
                else:
                    interpolated_values[i] = np.interp(i, [j for j in range(len(interpolated_values)) if interpolated_values[j] >= lower_bound and interpolated_values[j] <= upper_bound], [interpolated_values[j] for j in range(len(interpolated_values)) if interpolated_values[j] >= lower_bound and interpolated_values[j] <= upper_bound])
        return interpolated_values
    
    def rollingRMS(self, DataListIN, RollingSize =14): #verdier må endres her 

        """ cutoff = 5
        nyquist_freq = 0.5 * self.qrs_detector.signal_frequency
        normalCutoff = cutoff/nyquist_freq
        sos = butter(3, normalCutoff, btype='high', output='sos' )
        DataListIN = sosfiltfilt(sos, DataListIN)

        aveNumber = sum(DataListIN)/len(DataListIN)
        aveList = np.full_like(DataListIN, fill_value=aveNumber)
        DataList = DataListIN-aveList

        roll = int(RollingSize/2)
        returnList = np.empty_like(DataList)
        #padding = np.zeros( roll ) 
        aveS = (DataList[0]+DataList[1]+DataList[2]+DataList[3]+DataList[4]+DataList[5]) /6
        paddingstart = np.full(roll, fill_value=aveS)
        
        aveE = (DataList[-6]+DataList[-1]+DataList[-2]+DataList[-3]+DataList[-4]+DataList[-5]) /6
        paddingend = np.full(roll, fill_value=aveE)

        PaddedData = np.append((np.append(paddingstart, DataList)), paddingend)
        PaddedData=PaddedData**2

        for i in range(len(DataList)):
            for x in range(roll):
                computingList = []
                computingList.append((PaddedData[i+roll-x]))#**2)
                computingList.append((PaddedData[i+roll+x]))#**2)
            returnList[i] = m.sqrt(sum(computingList)/len(computingList))

        peaks, _ = find_peaks(returnList, prominence=0.001, distance=500)
        prominences = peak_prominences(returnList, peaks)[0][0]
        contour_heights = returnList[peaks] - prominences
        results_half = peak_widths(returnList, peaks, rel_height=0.5)[0][0] 
 """
        prominences = 1
        results_half = 1
        return prominences, results_half
    

    def respiratory_rate(self):
        peaks, _ = find_peaks(self.BCG_Filtered_lowpass *-1, distance=200, prominence=5)
        counter = 0
        list = []
        for i in peaks:
            counter +=1
            list.append(i)
            if counter == 10:

                min = np.min(list)
                max = np.max(list)
                minutes = ((max-min)/self.qrs_detector.signal_frequency) / 60 #antall minutter brukt pp 10 slag
                self.respiratoryRate.append((len(list)/minutes))
                counter = 0





        
        X = np.linspace(start=0, stop = len(self.BCG_Filtered_lowpass)/self.qrs_detector.signal_frequency /60 , num = len(self.respiratoryRate))
        plt.plot(X, self.respiratoryRate)
        plt.show




""" def replace_outliers_zscore(self, y_values_input, threshold=0.50):
        # calculate the z-score of each value
        y_values = y_values_input.copy()
        z_scores = np.abs((y_values - np.mean(y_values)) / np.std(y_values))

        # replace outliers with the mean of the neighboring values
        for i in range(len(y_values)):
            if z_scores[i] > threshold:
                left_neighbor = y_values[i-1] if i > 0 else y_values[i]
                right_neighbor = y_values[i+1] if i < len(y_values)-1 else y_values[i]
                if i == 0:
                    y_values[i] = right_neighbor
                elif i == len(y_values)-1:
                    y_values[i] = left_neighbor
                else:
                    y_values[i] = (left_neighbor + right_neighbor) / 2

        return y_values 
        
        
    def replace_outliers_zscore(self, y_values_input, threshold=0.50):
        # calculate the z-score of each value
        y_values = y_values_input.copy()
        z_scores = np.abs((y_values - np.mean(y_values)) / np.std(y_values))

        # replace outliers with the mean of the neighboring values
        for i in range(1, len(y_values)-1):
            if z_scores[i] > threshold and z_scores[i-1] <= threshold and z_scores[i+1] <= threshold:
                left_neighbor = y_values[i-1]
                right_neighbor = y_values[i+1]
                y_values[i] = (left_neighbor + right_neighbor) / 2

        return y_values

    
    def replace_outliers_avg(self, values, window_size=5, threshold=3):
            # Make a copy of the input list
            new_values = list(values)

            # Loop through the list and replace outliers with the average of neighboring values
            for i in range(len(values)):
                if i < window_size // 2 or i >= len(values) - window_size // 2:
                    continue
                window = values[i - window_size // 2 : i + window_size // 2 + 1]
                avg = sum(window) / len(window)
                if abs(values[i] - avg) > threshold:
                    new_values[i] = avg

            return new_values
            
        
        
        """


    

