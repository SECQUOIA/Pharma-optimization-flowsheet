"""Random instance generator for pharmaceutical optimization flowsheet.

This module generates random test data for pharmaceutical manufacturing optimization
problems, creating CSV files with manufacturing and transportation data.
"""

import random


class Generator:
    """Generates random manufacturing and transportation data for optimization models.

    This class creates synthetic data for multi-step pharmaceutical manufacturing
    processes with multiple vendor options at each step."""
    def __init__(self, steps: int, options: int):
        """Initialize the generator with problem dimensions.

        Args:
            steps: Number of manufacturing steps in the process.
            options: Number of vendor options available at each step.
        """
        self.steps = steps
        self.options = options

    def number_to_letters(self, n):
        """Convert a number to Excel-style column letters (1=A, 26=Z, 27=AA, etc.).

        Args:
            n: Positive integer to convert.

        Returns:
            String representation using letters (e.g., 1 -> 'A', 27 -> 'AA').
        """
        result = ""
        while n > 0:
            n -= 1
            result = chr(65 + (n % 26)) + result
            n //= 26
        return result
    
    def createManufacturingData(self):
        """Generate random manufacturing data and save to CSV file.

        Creates a CSV file 'manufacturingData2.csv' with columns:
        - step: Manufacturing step number (1 to steps)
        - Vendor: Vendor identifier (letter-based: A, B, C, ...)
        - Production: Random production capacity (1 to 99,999)
        - Cost: Random manufacturing cost (1 to 99,999)

        The file contains one row for each (step, vendor) combination.
        """
        with open("manufacturingData2.csv", 'w') as f:
            f.write("step,Vendor,Production,Cost\n")
            for step in range(1, self.steps + 1):
                for option in range(1, self.options + 1):
                    f.write(str(step) + ',' + self.number_to_letters(option) + ',' +
                            str(random.randrange(1, 100000)) + ',' + 
                            str(random.randrange(1, 100000)) + "\n")
    
    def createTransportData(self):
        """Generate random transportation data and save to CSV file.

        Creates a CSV file 'transportData2.csv' with columns:
        - Step1: Source manufacturing step number
        - Step2: Destination manufacturing step number (Step1 + 1)
        - Vendor1: Source vendor identifier (letter-based)
        - Vendor2: Destination vendor identifier (letter-based)
        - TransCost: Transportation cost (0 if same vendor, otherwise 1 to 100)

        The file contains one row for each possible transition between consecutive
        steps and vendor pairs. Transportation cost is 0 when the same vendor handles
        both steps (no physical transport needed).
        """
        with open("transportData2.csv", 'w') as f:
            f.write('Step1,Step2,Vendor1,Vendor2,TransCost\n')
            for step in range(1, self.steps):
                for option1 in range(1, self.options + 1):
                    for option2 in range(1, self.options + 1):
                        vendor1 = self.number_to_letters(option1)
                        vendor2 = self.number_to_letters(option2)
                        cost = 0 if vendor1 == vendor2 else random.randrange(1, 101)
                        f.write(f"{step},{step + 1},{vendor1},{vendor2},{cost}\n")
    