import GEOparse
import pandas as pd
import os

class GEODataManager:
    def __init__(self, gse_id):
        self.gse_id = gse_id
        self.gse = None

    def download_dataset(self):
        """Download the dataset if not already present in the local directory."""
        expected_file_path = f"{self.gse_id}_family.soft.gz"
        
        if not os.path.exists(expected_file_path):
            self.gse = GEOparse.get_GEO(geo=self.gse_id, destdir=".")
            print(f"Downloaded dataset: {self.gse_id}")
        else:
            self.gse = GEOparse.get_GEO(filepath=expected_file_path)
            print(f"Dataset {self.gse_id} is already downloaded and loaded from local file.")
        return self.gse
    
    def display_metadata(self):
        """Dynamically prints all available metadata of the GSE object."""
        for key, value in self.gse.metadata.items():
            print(f"{key.replace('_', ' ').title()}:")
            if isinstance(value, list) and len(value) > 1:
                for item in value:
                    print(f"- {item}")
            else:
                print(f"- {''.join(value)}")
            print()  # Print a newline for better readability
    
    @staticmethod
    def load_csv(filename, index_col=None):
        """
        Load a CSV file into a pandas DataFrame.
        
        Parameters:
        - filename: str, the path to the CSV file.
        - index_col: str or int, optional, default=None. Column to set as index. Use None for default index.
        
        Returns:
        - DataFrame: The loaded DataFrame.
        """
        try:
            df = pd.read_csv(filename, index_col=index_col)
            df.set_index('Unnamed: 0', inplace=True, drop=True)
            if df.index.name is not None and df.index.name != "":
                df.index.name = None

            print(f"CSV file '{filename}' successfully loaded.")
            return df
        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while loading the CSV file: {e}")
            return None

    @staticmethod
    def save_to_csv(dataframe, merged_file='GSE44077_merged_data.csv'):
        """Save the merged data and gene mappings to CSV files."""
        dataframe.to_csv(merged_file, index=True, header=True)
        print(f"Merged data saved to {merged_file}.")
    @staticmethod
    def clean_entrez_gene_id(merged_data, column_name='EntrezGeneID'):
        """Cleans merged data based on EntrezGeneID."""
        cleaned_df = merged_data.dropna(subset=[column_name])
        cleaned_df = cleaned_df[~cleaned_df[column_name].isin(["---", ""])]
        return cleaned_df

    @staticmethod
    def print_common_values_percentage(df1, df2, column1, column2):
        """Finds and prints the percentage of common values between two specified columns from two DataFrames."""
        print("total NAN in DataFrame 1 is: ", df1[column1].isna().sum())
        print("total NAN in DataFrame 2 is", df2[column2].isna().sum())
        unique_values_1 = set(df1[column1].dropna().unique())
        unique_values_2 = set(df2[column2].dropna().unique())

        common_values = unique_values_1.intersection(unique_values_2)
        total_common = len(common_values)
        
        percent_common = (total_common / min(len(unique_values_1), len(unique_values_2))) * 100
        
        print(f"Common values between {column1} and {column2}: {total_common}")
        print(f"Percentage of common values: {percent_common:.2f}%")


    def modify_dataframe(self,df, include_columns=None, drop_columns=None,index_column = None):
        """
        Modifies the input DataFrame based on included and dropped columns.
        
        Parameters:
        - df: pandas.DataFrame, the original DataFrame.
        - include_columns: list or None, columns to include in the final DataFrame.
        - drop_columns: list or None, columns to drop from the DataFrame.
        
        Returns:
        - pandas.DataFrame, the modified DataFrame.
        """
        if index_column :
            df.index = df[index_column]
        
        
        # If include_columns is specified and not empty, filter the DataFrame to include only these columns
        if include_columns:
            df = df[[col for col in include_columns if col in df.columns]]
        
        # Drop specified columns if drop_columns is not None or empty
        if drop_columns:
            df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')
        if df.index.name is not None and df.index.name != "":
            df.index.name = None
        return df

class DisplayOptionsManager:
    @staticmethod
    def toggle_pandas_display_options(enable):
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


class GEODataParser:
    def __init__(self, gse):
        self.gse = gse
        self.mapping_df = pd.DataFrame()
        self.expression_data = pd.DataFrame()

    def parse_gpl_data_gpl570(self):
        """Parse GPL570 data to extract mappings."""
        gpl_id = list(self.gse.gpls.keys())[0]
        gpl = self.gse.gpls[gpl_id] 

        # Check if the GPL is GPL570
        if gpl.name != "GPL570":
            print("This function is designed for GPL570.")
            return

        mappings = []
        for index, row in gpl.table.iterrows():
            probe_id = row['ID']
            gene_symbol = row['Gene Symbol'].strip() if pd.notnull(row['Gene Symbol']) else ''
            entrez_gene_id = str(row['ENTREZ_GENE_ID']).strip() if pd.notnull(row['ENTREZ_GENE_ID']) else ''

            mappings.append({
                'ProbeID': probe_id,
                'GeneSymbol': gene_symbol,
                'EntrezGeneID': entrez_gene_id
            })


        self.mapping_df = pd.DataFrame(mappings).drop_duplicates(subset='ProbeID', keep='first')
        print("Parsed GPL570 data and filtered for unique mappings.")
        return self.mapping_df

    def parse_gpl_data(self):
        """Parse GPL data to extract mappings."""
        gpl_id = list(self.gse.gpls.keys())[0]
        gpl = self.gse.gpls[gpl_id] 

        if gpl.name == "GPL570":
            gpl570_df = self.parse_gpl_data_gpl570()
            return gpl570_df
        else:
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
            return self.mapping_df

    def compile_expression_data(self):
        """Compile expression data from all samples."""
        for gsm_name, gsm in self.gse.gsms.items():
            df = pd.DataFrame(gsm.table).set_index('ID_REF')
            self.expression_data[gsm_name] = df['VALUE']
        self.expression_data.reset_index(inplace=True)
        print("Compiled expression data from all samples.")
        return self.expression_data
    
    def add_labels_to_expression_data(self, expression_data, metadata_key='characteristics_ch1', label_prefix=None, separator=':'):
        """Add sample labels to the expression data based on a given DataFrame."""
        labels = {}
        for gsm_name, gsm in self.gse.gsms.items():
            # Accessing the specified metadata
            metadata_values = gsm.metadata.get(metadata_key, [])
            label = None
            for value in metadata_values:
                # Check if label_prefix is specified and is part of the metadata value
                if label_prefix and value.startswith(label_prefix):
                    # Split the value by the separator and extract the label
                    parts = value.split(separator)
                    if len(parts) > 1:
                        label = parts[1].strip()
                        break
            labels[gsm_name] = label if label else 'Unknown'  # Use 'Unknown' or another placeholder if no label is found

        # Convert labels dict to DataFrame and merge it with the expression data
        labels_df = pd.DataFrame.from_dict(labels, orient='index', columns=['Label']).reset_index()
        labels_df.rename(columns={'index': 'Sample'}, inplace=True)
        
        # Assuming the expression data's columns are GSM names, transpose it before merging
        expression_data_transposed = expression_data.transpose()
        sample_to_label_map = labels_df.set_index('Sample')['Label'].to_dict()

        # Map the labels to the index of expression_data_transposed and create a new column for labels
        expression_data_transposed['Label'] = expression_data_transposed.index.map(sample_to_label_map)

        # Now expression_data_transposed has a new column 'Label' with the corresponding labels
        print(expression_data_transposed.head(3))        
        print("Added labels to the expression data.")
        return expression_data_transposed

class GEODataMerger:
    def __init__(self, expression_data, mapping_df):
        self.expression_data = expression_data
        self.mapping_df = mapping_df
        self.merged_data = pd.DataFrame()

    def merge_data(self):
        """Merge expression data with gene mappings."""
        self.merged_data = pd.merge(self.expression_data, self.mapping_df, left_on='ID_REF', right_on='ProbeID', how='left')
        print("Merged expression data and gene mappings.")
        return self.merged_data


# # Example usage
# if __name__=="main":


#     gse_id = "GSE44077"
#     manager  = GEODataManager(gse_id=gse_id)
#     df = manager.load_csv('final_data.csv')


#     DisplayOptionsManager.toggle_pandas_display_options(False)

#     gse_id = "GSE44077"
#     manager  = GEODataManager(gse_id=gse_id)
#     gse = manager.download_dataset()
#     manager.display_metadata()
#     parser = GEODataParser(gse=gse)
#     mapping_df = parser.parse_gpl_data()
#     expression_data = parser.compile_expression_data()
#     merger = GEODataMerger(expression_data=expression_data, mapping_df=mapping_df)
#     merged_data = merger.merge_data()

#     cleaned_data = manager.clean_entrez_gene_id(merged_data=merged_data, column_name='EntrezGeneID')

#     cleaned_data = manager.modify_dataframe(cleaned_data,None,['ProbeID',"ID_REF",
#         'Ref_DB', 'Location', 'GeneSymbol', 'ChromosomalLocation',"EntrezGeneID"],"EntrezGeneID")

#     final_data = parser.add_labels_to_expression_data(cleaned_data , metadata_key='characteristics_ch1', label_prefix='tissue', separator=':')


#     manager.save_to_csv(dataframe=final_data, merged_file="final_data.csv")
