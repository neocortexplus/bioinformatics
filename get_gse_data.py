import GEOparse
import pandas as pd
import os 


class GEODataHandler:
    def __init__(self, gse_id):
        self.gse_id = gse_id
        self.gse = None
        self.expression_data = pd.DataFrame()
        self.mapping_df = pd.DataFrame()
        self.merged_data = pd.DataFrame()
        self.cleaned_df = pd.DataFrame()

    def toggle_pandas_display_options(enable=True):
        """Toggle enhanced pandas display options based on the 'enable' flag."""
        if enable:
            pd.set_option('display.max_columns', None)  # Show all columns
            pd.set_option('display.max_rows', 10)  # Adjust the number of rows to display as needed
            pd.set_option('display.max_colwidth', None)  # Show full content of each column
            pd.set_option('display.width', None)  # Adjust display width to show each row on one line, depending on your display
        else:
            # Reset to default
            pd.reset_option('display.max_columns')
            pd.reset_option('display.max_rows')
            pd.reset_option('display.max_colwidth')
            pd.reset_option('display.width')

    def download_dataset(self):
        """Download the dataset if not already present in the local directory."""
        # Define the expected file path for the dataset
        expected_file_path = f"{self.gse_id}_family.soft.gz"
        
        # Check if the file already exists
        if not os.path.exists(expected_file_path):
            # Download the dataset if the file does not exist
            self.gse = GEOparse.get_GEO(geo=self.gse_id, destdir=".")
            print(f"Downloaded dataset: {self.gse_id}")
        else:
            # Load the dataset from the local file if it exists
            self.gse = GEOparse.get_GEO(filepath=expected_file_path)
            print(f"Dataset {self.gse_id} is already downloaded and loaded from local file.")

    def parse_gpl_data(self):
        """Parse GPL data to extract mappings."""
        gpl_id = list(self.gse.gpls.keys())[0]
        gpl = self.gse.gpls[gpl_id]
        
        mappings = []
        for index, row in gpl.table.iterrows():
            probe_id = row['ID']
            assignments = row['gene_assignment'].split('///')
            
            for assignment in assignments:
                parts = assignment.split('//')
                
                # Initialize variables to store information, set defaults to an empty string
                ref_db = ''
                location = ''
                gene_symbol = ''
                chromosomal_locations = ''
                entrez_gene_id = ''
                
                # Check the number of parts and assign accordingly
                if len(parts) >= 1:
                    ref_db = parts[0].strip()
                if len(parts) >= 2:
                    location = parts[1].strip()
                if len(parts) >= 3:
                    gene_symbol = parts[2].strip()
                if len(parts) >= 4:
                    chromosomal_locations = parts[3].strip()
                if len(parts) >= 5:
                    entrez_gene_id = parts[4].strip()
                
                mappings.append({
                    'ProbeID': probe_id,
                    'Ref_DB': ref_db,
                    'Location': location,
                    'GeneSymbol': gene_symbol,
                    'ChromosomalLocation': chromosomal_locations,
                    'EntrezGeneID': entrez_gene_id
                })

        self.mapping_df = pd.DataFrame(mappings).drop_duplicates(subset='ProbeID', keep='first')
        print("Parsed GPL data and filtered for unique mappings.")

    def compile_expression_data(self):
        """Compile expression data from all samples."""
        for gsm_name, gsm in self.gse.gsms.items():
            df = pd.DataFrame(gsm.table).set_index('ID_REF')
            self.expression_data[gsm_name] = df['VALUE']
        self.expression_data.reset_index(inplace=True)
        print("Compiled expression data from all samples.")
    
    def merge_data(self):
        """Merge expression data with gene mappings."""
        self.merged_data = pd.merge(self.expression_data, self.mapping_df, left_on='ID_REF', right_on='ProbeID', how='left')
        print("Merged expression data and gene mappings.")


    def clean_entrez_gene_id(self,column_name='EntrezGeneID'):
        """
        Removes rows from a DataFrame where the EntrezGeneID column contains NaN, "---", "", or other invalid values.
        
        Parameters:
        - df: The pandas DataFrame to be cleaned.
        - column_name: The name of the column to check for invalid values (default is 'EntrezGeneID').
        
        Returns:
        - A pandas DataFrame with the invalid rows removed.
        """
        # Remove rows where the column is NaN
        self.cleaned_df = self.merged_data.dropna(subset=[column_name])
        
        # Further remove rows where the column value is "---" or an empty string ""
        self.cleaned_df = self.cleaned_df[~self.cleaned_df[column_name].isin(["---", ""])]
        
        return self.cleaned_df    
    def save_to_csv(self, merged_file='GSE44077_merged_data.csv', mapping_file='GSE44077_gene_mapping.csv'):
        """Save the merged data and gene mappings to CSV files."""
        self.merged_data.to_csv(merged_file, index=False)
        self.mapping_df.to_csv(mapping_file, index=False)
        print(f"Merged data saved to {merged_file}.")
        print(f"Gene mapping saved to {mapping_file}.")

# Example usage
gse_id = "GSE44077"
handler = GEODataHandler(gse_id=gse_id)
handler.download_dataset()
handler.parse_gpl_data()
handler.compile_expression_data()
handler.merge_data()
handler.toggle_pandas_display_options()
handler.clean_entrez_gene_id('EntrezGeneID')
handler.save_to_csv()
