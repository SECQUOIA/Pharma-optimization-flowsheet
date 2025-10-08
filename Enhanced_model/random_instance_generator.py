"""Random instance generator for pharmaceutical optimization flowsheet.

This module generates random test data for pharmaceutical manufacturing optimization
problems, creating CSV files with manufacturing and transportation data.
"""

import random
import pandas as pd
from openpyxl.utils import get_column_letter


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
        return get_column_letter(n)

    def saveDataToCSV(self, df: pd.DataFrame, filename: str):
        """Save a DataFrame to a CSV file.

        Args:
            df: The pandas DataFrame to save.
            filename: The name of the output CSV file (with or without .csv extension).
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
        df.to_csv(filename, index=False)

    def generateManufacturingData(self) -> pd.DataFrame:
        """Generate random manufacturing data and return as DataFrame.

        Returns:
            DataFrame with columns:
            - step: Manufacturing step number (1 to steps)
            - Vendor: Vendor identifier (letter-based: A, B, C, ...)
            - Production: Random production capacity (1 to 99,999)
            - Cost: Random manufacturing cost (1 to 99,999)

            Contains one row for each (step, vendor) combination.
        """
        data = []
        for step in range(1, self.steps + 1):
            for option in range(1, self.options + 1):
                data.append({
                    'step': step,
                    'Vendor': self.number_to_letters(option),
                    'Production': random.randrange(1, 100000),
                    'Cost': random.randrange(1, 100000)
                })
        return pd.DataFrame(data)

    def generateTransportData(self) -> pd.DataFrame:
        """Generate random transportation data and return as DataFrame.

        Returns:
            DataFrame with columns:
            - SourceStep: Source manufacturing step number
            - DestinationStep: Destination manufacturing step number (SourceStep + 1)
            - SourceVendor: Source vendor identifier (letter-based)
            - DestinationVendor: Destination vendor identifier (letter-based)
            - TransportationCost: Transportation cost (0 if same vendor, otherwise 1 to 100)

            Contains one row for each possible transition between consecutive steps and
            vendor pairs. Transportation cost is 0 when the same vendor handles both steps.
        """
        data = []
        for step in range(1, self.steps):
            for option1 in range(1, self.options + 1):
                for option2 in range(1, self.options + 1):
                    vendor1 = self.number_to_letters(option1)
                    vendor2 = self.number_to_letters(option2)
                    cost = 0 if vendor1 == vendor2 else random.randrange(1, 101)
                    data.append({
                        'SourceStep': step,
                        'DestinationStep': step + 1,
                        'SourceVendor': vendor1,
                        'DestinationVendor': vendor2,
                        'TransportationCost': cost
                    })
        return pd.DataFrame(data)

    def createManufacturingData(self, filename: str = "manufacturingData2.csv"):
        """Generate random manufacturing data and save to CSV file.

        Args:
            filename: Name of the output CSV file (default: "manufacturingData2.csv")

        Creates a CSV file with columns:
        - step: Manufacturing step number (1 to steps)
        - Vendor: Vendor identifier (letter-based: A, B, C, ...)
        - Production: Random production capacity (1 to 99,999)
        - Cost: Random manufacturing cost (1 to 99,999)

        The file contains one row for each (step, vendor) combination.
        """
        df = self.generateManufacturingData()
        self.saveDataToCSV(df, filename)

    def createTransportData(self, filename: str = "transportData2.csv"):
        """Generate random transportation data and save to CSV file.

        Args:
            filename: Name of the output CSV file (default: "transportData2.csv")

        Creates a CSV file with columns:
        - SourceStep: Source manufacturing step number
        - DestinationStep: Destination manufacturing step number (SourceStep + 1)
        - SourceVendor: Source vendor identifier (letter-based)
        - DestinationVendor: Destination vendor identifier (letter-based)
        - TransportationCost: Transportation cost (0 if same vendor, otherwise 1 to 100)

        The file contains one row for each possible transition between consecutive
        steps and vendor pairs. Transportation cost is 0 when the same vendor handles
        both steps (no physical transport needed).
        """
        df = self.generateTransportData()
        self.saveDataToCSV(df, filename)
    