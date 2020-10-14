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
        return {'createDataSet': 0, 'autoBuild': 1 , 'model' : 2, 'loadDataSet' : 3}.get(string_input, -1)

    @classmethod
    def _secondCase(cls,string_input):
        return {'train': 0, 'test': 1 }.get(string_input, -1)

    @classmethod
    def _dataSet(cls,array_input):
        thirdCase = -1
        if len(array_input) < 3:
            cls.errorOccur("It is short command","type int only (what > 0) ")
        else:
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
    def _loadDataSet(cls,array_input):
        return cls._createDataSet(array_input)

    @classmethod
    def _autoBuild(cls,array_input):
        cds = command([])
        if len(array_input) == 1:
            pass
        else:
            string_input = array_input[1]
            try:
                int_input = int(string_input)
                if int_input < 0:
                    cls.errorOccur(int_input,"type int only (what >= 0) ")
                else:
                    cds._extend(command([int_input]))
            except:
                cls.errorOccur(string_input,"type int only (what >= 0) ")

        if len(array_input) > 2:
            string_input = array_input[2]
            try:
                int_input = int(string_input)
                if int_input < 0:
                    cls.errorOccur(int_input,"type int only (what >= 0) ")
                else:
                    cds._extend(command([int_input]))
            except:
                cls.errorOccur(string_input,"type int only (what >= 0) ")
        return cds

    @classmethod
    def _model(cls,array_input):
        cds = command([])
        if len(array_input) < 2:
            cls.errorOccur("It is short command","train, test")
        else:
            cds.commands = [cls._secondCase(array_input[1])]
        return cds

    @classmethod
    def userInput(cls,string_input):
        cls.result.noErr = True
        if len(string_input) > 0:
            array_input = string_input.strip().split()
            firstCase = cls._firstCase(array_input[0])
            cls.result.commands = [firstCase]

            if firstCase < 0:
                cls.errorOccur(array_input[0], "createDataSet, autoBuild, model, loadDataSet")
            cls.result.command = firstCase

            if firstCase == 0:
                secondCase = cls._createDataSet(array_input)

            elif firstCase == 1:
                secondCase = cls._autoBuild(array_input)

            elif firstCase == 2:
                secondCase = cls._model(array_input)

            elif firstCase == 3:
                secondCase = cls._loadDataSet(array_input)

            try:
                cls.result._extend(secondCase)
            except:
                print("  err : UNKNOWN")
                cls.result.noErr = False
        else:   cls.errorOccur("", "createDataSet, autoBuild, model, loadDataSet")
        return cls.result