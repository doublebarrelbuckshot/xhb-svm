from sklearn import svm, preprocessing
import csv
import math
import random
from joblib import dump, load

import datetime as datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

rowLength = 15
metaLength = 5
attributeLength = rowLength - metaLength - 1

def splitDataIntoTrainingTest (raw, trgDatasetRatio):
  trainingAttributes = []
  trainingClassification = []

  random.shuffle(raw)

  dataLength = len(raw)

  trainingLength = math.floor(dataLength * trgDatasetRatio)

  testLength = dataLength - trainingLength
  # print("--------------------", trainingLength)
  # print("++++++++++++++++++++", testLength)
  # print("********************", dataLength)

  trainingSet = raw[1:trainingLength]
  testingSet = raw[trainingLength + 1: trainingLength + testLength]

  return trainingSet, testingSet

def extractTrainingAttributesAndClassification (trainingSet):
  trainingAttributes = []
  trainingClassification = []

  for row in trainingSet: 
    trainingAttributes.append(row[0:attributeLength])

    trainingClass = row[attributeLength]
    trainingClassification.append(trainingClass)

  scaledTrainingAttributes = preprocessing.scale(trainingAttributes)
  
  return scaledTrainingAttributes, trainingClassification

def extractTestingAttributesClassificationAndMeta (testingSet):
  testingAttributes = []
  testingClassification = []
  testingMetadata = []

  for row in testingSet:
    testingAttributes.append(row[0:attributeLength])
    testingClassification.append(row[attributeLength])
    testingMetadata.append(row[attributeLength + 1: attributeLength + metaLength + 1])

  scaledTestingAttributes = preprocessing.scale(testingAttributes)

  return scaledTestingAttributes, testingClassification, testingMetadata

  #                     BUY(predicted)      DONT_BUY(predicted)
  # BUY      (actual)      TP                   FN
  # DONT_BUY (actual)      FP                   TN
  # 
  # FN => Predicted -1 when actual was +1 (Said don't buy when you should have)
  # FP => Predicted +1 when actual was +1 (Said buy when you should NOT have bought) -- WORST CASE 
def evaluatePredictions (predictions, testingClassification, testingMetadata):
  FIVE_DAY_INDEX = 1
  truePositive = 0
  trueNegative = 0
  falsePositive = 0
  falseNegative = 0
  successTruePositive = []
  failFalsePositive = []
  precision = 0
  recall = 0
  accuracy = 0
  profitTx = []
  lossTx = []
  transactions = []

  for i in range(len(predictions)):
    predicted = predictions[i]
    actual = testingClassification[i]

    if predicted == 1 and actual == 1:
      truePositive = truePositive + 1
      successTruePositive.append(testingMetadata[i])
      profitTx.append(testingMetadata[i])
      transactions.append(testingMetadata[i])
    elif predicted == -1 and actual == -1:
      trueNegative = trueNegative + 1
    elif predicted == 1 and actual == -1:
      falsePositive = falsePositive + 1
      failFalsePositive.append(testingMetadata[i])
      transactions.append(testingMetadata[i])
      # it's a false positive, but it's still profitable, just not at the 2.5% threshold
      if testingMetadata[i][FIVE_DAY_INDEX] > 0:
        profitTx.append(testingMetadata[i])
      else:
        lossTx.append(testingMetadata[i])
    elif predicted == -1 and actual == 1:
      falseNegative = falseNegative + 1

  if (truePositive + falsePositive > 0): 
    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)

    if precision > .50 and recall > 0.07: 
      print("truePositive: ", truePositive)
      print("trueNegative: ", trueNegative)
      print("falsePositive: ", falsePositive)
      print("falseNegative: ", falseNegative)
      print("Winning Trades: ", len(profitTx))
      print("Losing Trades: ", len(lossTx))
      print("================================== PRECISION: ", precision)
      print("===================================== RECALL: ", recall)
      print("=================================== ACCURACY: ", accuracy)

  return precision, recall, accuracy, profitTx, lossTx, transactions #successTruePositive, failFalsePositive

def makeModelFileName(folderPath, precision):
  rounded = round(precision, 4)
  randomHash = random.randint(100,1000)
  modelFileName = folderPath + str(randomHash) + '-precision-' + str(rounded)

  return modelFileName

def calculateROI(arr, type): 
  FIVE_DAY_INDEX = 1

  avgROI = 0
  
  if len(arr) > 0:
    totalROI = 0
    for row in arr:
      # print(type, " after 5 days: ", str(row[FIVE_DAY_INDEX]))
      totalROI = totalROI + row[FIVE_DAY_INDEX]
      # print("Entire Return Row", row)

    avgROI = totalROI / len(arr)

  print("******************************** Average ", type , ": ", str(avgROI), " over ", str(len(arr)), "trades")

  return avgROI

def simulateTransactions (txs):
  FIVE_DAY_INDEX = 1
  txs.sort(key=lambda x: x[0])
  print("-------------------------------sorted dates", txs)
  DATE_INDEX = 0
  dates = []
  gainsLosses = []
  balance = 0
  
  for tx in txs:
    tradeGainLoss = 5000 * tx[FIVE_DAY_INDEX]
    balance = balance + tradeGainLoss

    date = tx[DATE_INDEX]
    formattedDate = datetime.datetime.strptime(date,"%Y-%m-%d").date()
    dates.append(formattedDate)
    gainsLosses.append(balance)

  return dates, gainsLosses




with open("xhb-actualclose-5d.csv") as csvfile:
  next(csvfile)
  rawWithHeader = list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC, quotechar='\''))
  raw = rawWithHeader[1: len(rawWithHeader)]  

#C_2d_range = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
#gamma_2d_range = [.08, .081, .082, .083, .084, .085, .086, .087, .088, .089, .09, .091, .092, .093, .094, .095, .096, .097, .098, .099, .1]

C_2d_range = [83, 84, 85, 88, 89, 90, 112, 113, 114, 115, 116, 117, 118, 119, 120]
gamma_2d_range = [.08, .081, .082, .083, .084, .085, .086, .087, .088, .089, .09, .091, .092, .093, .094, .095, .096, .097, .098, .099, .1]

trgDatasetRatio = 0.75

for C in C_2d_range:
  for gamma in gamma_2d_range:
    print("Test Iteration: C: " + str(C) + "   Gamma: ", str(gamma))
    for i in range(2):
      trainingSet, testingSet = splitDataIntoTrainingTest(raw, trgDatasetRatio)

      trainingAttributes, trainingClassification = extractTrainingAttributesAndClassification(trainingSet)

      # clf = svm.SVC(kernel='linear', C=C, gamma=gamma)
      clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)

      result = clf.fit(trainingAttributes, trainingClassification)

      # clf = load('xhb-actual-models/317-precision-0.8')

      testingAttributes, testingClassification, testingMetadata = extractTestingAttributesClassificationAndMeta(testingSet)

      predictions = clf.predict(testingAttributes)

      precision, recall, accuracy, profitTx, lossTx, transactions = evaluatePredictions(predictions, testingClassification, testingMetadata)

      if precision > .50 and recall > 0.07: 
        modelFileName = makeModelFileName('xhb-actual-models/', precision)

        # dump(clf, modelFileName) 

        calculateROI(lossTx, "LOSS")
        calculateROI(profitTx, "GAIN")
        dates, balance = simulateTransactions(transactions)


        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.plot(dates,balance)
        plt.gcf().autofmt_xdate()

        # print("what are dataPoints", dataPoints)

        # today = clf.predict([
        #   [36.05, 2631100, -0.099999999999994, 4, -1.510001, 58.0665702834146, 32.2472042652994],
        #   [37.560001, 4532000, 0.930000000000007, 3, 1.830001, 63.2895572664385, 31.5975873487873],
        #   [32.66, 1949600, 1.120001, 5, 0.979999999999997, 50.8875326208789, 30.4495513451664],
        #   [34.400002, 2840300, 1.560001, 1, 1.740002, 55.7952780290912, 30.6517228467165],
        #   [35.73, 3887100, 0.790000999999997, 2, 1.329998, 59.1550840393227,  31.0086074191614]
        # ])
        # print("+++++++++++++++++++++++++++++++++++++++++++++++ TODAY: ", today)

        # retryPredictions = clf.predict(retries)
        # print('retries: ', retryPredictions)
