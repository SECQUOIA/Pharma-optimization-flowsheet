import random

class Generator:
    def __init__(self, steps: int, options: int):
        self.steps = steps
        self.options = options
    
    def number_to_letters(self, n):
        result = ""
        while n > 0:
            n -= 1
            result = chr(65 + (n % 26)) + result
            n //= 26
        return result
    
    def createManufacturingData(self):
        with open("manufacturingData2.csv", 'w') as f:
            f.write("step,Vendor,Production,Cost\n")
            for step in range(1, self.steps + 1):
                for option in range(1, self.options + 1):
                    f.write(str(step) + ',' + self.number_to_letters(option) + ',' +
                            str(random.randrange(1, 100000)) + ',' + 
                            str(random.randrange(1, 100000)) + "\n")
    
    def createTransportData(self):
        with open("transportData2.csv", 'w') as f:
            f.write('Step1,Step2,Vendor1,Vendor2,TransCost\n')
            for step in range(1, self.steps):
                for option1 in range(1, self.options + 1):
                    for option2 in range(1, self.options + 1):
                        vendor1 = self.number_to_letters(option1)
                        vendor2 = self.number_to_letters(option2)
                        cost = 0 if vendor1 == vendor2 else random.randrange(1, 101)
                        f.write(f"{step},{step + 1},{vendor1},{vendor2},{cost}\n")
    