class command:
    commands = []
    noErr = True
    def __init__(self, commands):
        self.commands = commands
    def _extend(self, cds):
        self.commands += cds.commands

class Decision:
    result = command([])
    @classmethod
    def errorOccur(cls,string_input,canUse):
        print("  err : Unknown command  ", string_input)
        print("  you can use :  ",canUse)
        cls.result.noErr = False
    @classmethod
    def _firstCase(cls,string_input): 
        return {'createDataSet': 0, 'autoBuild': 1 }.get(string_input, -1)

    @classmethod
    def _secondCase(cls,string_input):
        return {'train': 0, 'test': 1 }.get(string_input, -1)

    @classmethod
    def _dataSet(cls,array_input):
        if len(array_input) < 3:
            cls.errorOccur("It is short command","type int only (what > 0) ")
        try:
            thirdCase = int(array_input[2])
            if thirdCase < 1:
                cls.errorOccur(thirdCase,"type int only (what > 0) ")
        except:
            cls.errorOccur(array_input[2],"type int only (what > 0) ")
        return thirdCase

    @classmethod
    def _createDataSet(cls,array_input):
        try:
            secondCase = cls._secondCase(array_input[1])
        except:
            cls.errorOccur("It is short command","train, test")
        if cls.result.noErr:
            if secondCase < 0:
                cls.errorOccur(array_input[1],"train, test")
            else:
                thirdCase = cls._dataSet(array_input)
            return command([secondCase, thirdCase])

    @classmethod
    def userInput(cls,string_input):
        cls.result.noErr = True
        array_input = string_input.strip().split()
        firstCase = cls._firstCase(array_input[0])
        cls.result.commands = [firstCase]

        if firstCase < 0:
            cls.errorOccur(array_input[0], "createDataSet, autoBuild")
        cls.result.command = firstCase

        if firstCase == 0:
            try:
                cls.result._extend(cls._createDataSet(array_input))
            except:
                print("  err : UNKNOWN")
                cls.result.noErr = False

        return cls.result