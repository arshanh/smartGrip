import pickle
import pandas as pd
import sys
import json

class Hold:

    def __init__(self, data):

        # Data, certainty, and type of the hold
        self.data = data
        self.cert = -1
        self.type = ""

        # Raw count to determine which hold and certainty
        self.crimpCount = 0
        self.jugCount = 0
        self.miniJugCount = 0
        self.pinchCount = 0
        self.pocketCount = 0
        self.slopeCount = 0

    def predictType(self, scaler, mlp):
        scaledData = scaler.transform(self.data)
        predicts = mlp.predict(scaledData)

        for i in range(len(predicts)):
            if predicts[i][0] == 1:
                self.crimpCount += 1
            elif predicts[i][1] == 1:
                self.jugCount += 1
            elif predicts[i][2] == 1:
                self.miniJugCount += 1
            elif predicts[i][3] == 1:
                self.pinchCount += 1
            elif predicts[i][4] == 1:
                self.pocketCount += 1
            elif predicts[i][5] == 1:
                self.slopeCount += 1

        curMax = max(self.crimpCount, self.jugCount, self.miniJugCount, self.pinchCount, self.pocketCount,
                     self.slopeCount)

        if curMax == self.crimpCount:
            self.type = "Crimp"
        elif curMax == self.jugCount:
            self.type = "Jug"
        elif curMax == self.miniJugCount:
            self.type = "Mini Jug"
        elif curMax == self.pinchCount:
            self.type = "Pinch"
        elif curMax == self.pocketCount:
            self.type = "Pocket"
        elif curMax == self.slopeCount:
            self.type = "Slope"

        # Determined by the overall percentage of the the points that are classified as the selected grip
        self.cert = curMax / len(predicts)

# represents a single climb
class Climb:

    def __init__(self):
        # Count, array and certainty of crimp holds
        self.crimpCount = 0
        self.crimpHolds = []
        self.crimpCert = 0

        # Count, array and certainty of jug holds
        self.jugCount = 0
        self.jugHolds = []
        self.jugCert = 0

        # Count, array and certainty of mini jug holds
        self.miniJugCount = 0
        self.miniJugHolds = []
        self.miniJugCert = 0

        # Count, array and certainty of pinch holds
        self.pinchCount = 0
        self.pinchHolds = []
        self.pinchCert = 0

        # Count, array and certainty of pocket holds
        self.pocketCount = 0
        self.pocketHolds = []
        self.pocketCert = 0

        # Count, array and certainty of slope holds
        self.slopeCount = 0
        self.slopeHolds = []
        self.slopeCert = 0

    # Print the results of the climb
    def printClimb(self):

        print("")
        print("Climb results:")
        print("                      #  certainty")
        if self.crimpCount == 0:
            print("     Crimp Holds:     0  (N/A)")
        else:
            print("     Crimp Holds:     {}  ({:.3f})".format(self.crimpCount, self.crimpCert/self.crimpCount))
        if self.jugCount == 0:
            print("     Jug Holds:       0  (N/A)")
        else:
            print("     Jug Holds:       {}  ({:.3f})".format(self.jugCount, self.jugCert/self.jugCount))
        if self.miniJugCount == 0:
            print("     Mini Jug Holds:  0  (N/A)")
        else:
            print("     Mini Jug Holds:  {}  ({:.3f})".format(self.miniJugCount, self.miniJugCert/self.miniJugCount))
        if self.pinchCount == 0:
            print("     Pinch Holds:     0  (N/A)")
        else:
            print("     Pinch Holds:     {}  ({:.3f})".format(self.pinchCount, self.pinchCert/self.pinchCount))
        if self.pocketCount == 0:
            print("     Pocket Holds:    0  (N/A)")
        else:
            print("     Pocket Holds:    {}  ({:.3f})".format(self.pocketCount, self.pocketCert/self.pocketCount))
        if self.slopeCount == 0:
            print("     Slope Holds:     0  (N/A)")
        else:
            print("     Slope Holds:     {}  ({:.3f})".format(self.slopeCount, self.slopeCert/self.slopeCount))

    # Add a crimp hold
    def addCrimp(self, hold):

        self.crimpCount += 1
        self.crimpHolds.append(hold)
        self.crimpCert += hold.cert

    # Add a Jug hold
    def addJug(self, hold):

        self.jugCount += 1
        self.jugHolds.append(hold)
        self.jugCert += hold.cert

    # Add a Mini Jug hold
    def addMiniJug(self, hold):

        self.miniJugCount += 1
        self.miniJugHolds.append(hold)
        self.miniJugCert += hold.cert

    # Add a Pinch hold
    def addPinch(self, hold):

        self.pinchCount += 1
        self.pinchHolds.append(hold)
        self.pinchCert += hold.cert

    # Add a Pocket hold
    def addPocket(self, hold):

        self.pocketCount += 1
        self.pocketHolds.append(hold)
        self.pocketCert += hold.cert

    # Add a Slope hold
    def addSlope(self, hold):

        self.slopeCount += 1
        self.slopeHolds.append(hold)
        self.slopeCert += hold.cert


def main():

    if len(sys.argv) != 3:
        print("Usage:", sys.argv[0], "inputFolderName numberOfFiles")

    # Import neural network
    filename = "nn.lib"
    mlp = pickle.load(open(filename, 'rb'))

    file = open("sc.lib", 'rb')
    scaler = pickle.load(file)
    # print(type(scaler))

    climb = Climb()

    # Interrate through the filtered data files and add each grip to the climb
    for i in range(int(sys.argv[2])):
        data = pd.read_csv(sys.argv[1]+"/filt"+str(i)+".csv", header=None)
        # data = pd.read_csv(sys.argv[1]+"\\grip"+str(i)+".csv")
        temp_hold = Hold(data)
        temp_hold.predictType(scaler, mlp)

        if temp_hold.type == "Crimp":
            climb.addCrimp(temp_hold)
        elif temp_hold.type == "Jug":
            climb.addJug(temp_hold)
        elif temp_hold.type == "Mini Jug":
            climb.addMiniJug(temp_hold)
        elif temp_hold.type == "Pinch":
            climb.addPinch(temp_hold)
        elif temp_hold.type == "Pocket":
            climb.addPocket(temp_hold)
        elif temp_hold.type == "Slope":
            climb.addSlope(temp_hold)
        else:
            print("Unrecognized Hold")

    climb.printClimb()
    with open("predictions.json", "w") as outfile:
        json.dump(climb, fp=outfile, indent=4, sort_keys=True, default=lambda o: o.__dict__)

    # inputs = pd.read_csv("inputs.csv", header=None) # REPLACE with below...
    # inputs = pd.read_csv(argv[1]+String(i)+".csv", header=NONE)

    # inputs = scaler.transform(inputs)

    # outputs = mlp.predict(inputs)
    # output = mlp.predict(inputs[50].reshape)

    # predictions = [0 for i in range(len(outputs))]

    # for i in range(len(outputs)):
    #     if outputs[i][0] == 1:
    #         predictions[i] = "crimp"
    #     elif outputs[i][1] == 1:
    #         predictions[i] = "jug"
    #     elif outputs[i][2] == 1:
    #         predictions[i] = "Mini Jug"
    #     elif outputs[i][3] == 1:
    #         predictions[i] = "Pinch"
    #     elif outputs[i][4] == 1:
    #         predictions[i] = "Pocket"
    #     elif outputs[i][5] == 1:
    #         predictions[i] = "slope"

    # print("Prediction:", *predictions, sep='\n')


if __name__ == "__main__":
    main()
