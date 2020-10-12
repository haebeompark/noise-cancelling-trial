class command:
    commands = []
    def __init__(self, commands):
        self.commands = commands
    def _extend(self, commands):
        self.commands = self.commands.extend(commands)

class Decision:
    @classmethod
    def _firstCase(cls,string_input): 
        return {'createDataSet': 0, 'autoBuild': 1 }.get(string_input, -1)

    @classmethod
    def _secondCase(cls,string_input):
        return {'trainSet': 0, 'testSet': 1 }.get(string_input, -1)

    @classmethod
    def _dataSet(cls,string_input):
        try:
            thirdCase = int(string_input)
        except:
            print("  err : type mismatch  ", string_input)
            print("  you can use :  type int only (what > 0) ")
        return thirdCase

    @classmethod
    def _createDataSet(cls,array_input):
        secondCase = cls._secondCase(array_input[1])
        if secondCase < 0:
            print("  err : Unknown command  ", array_input[1])
            print("  you can use :  trainSet, testSet")
        else:
            thirdCase = cls._dataSet(array_input[2])

        return command([secondCase, thirdCase])

    @classmethod
    def userInput(cls,string_input):
        array_input = string_input.strip().split()
        firstCase = cls._firstCase(array_input[0])

        if firstCase < 0:
            print("  err : Unknown command  ", array_input[0])
            print("  you can use :  createDataSet, autoBuild")
        result = command(firstCase)

        if firstCase == 0:
            result._extend(_createDataSet(array_input))


        return result