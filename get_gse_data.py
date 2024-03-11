import GEOparse
import pandas as pd

# Step 1: Download the dataset
gse_id = "GSE44077"
gse = GEOparse.get_GEO(geo=gse_id)



# Step 2: Access the GPL data
gpl_id = list(gse.gpls.keys())[0]  # Assuming one GPL file; adjust as needed
gpl = gse.gpls[gpl_id]

mappings = []  # Initialize an empty list to store mappings

for index, row in gpl.table.iterrows():
    probe_id = row['ID']
    assignments = row['gene_assignment'].split('///')  # Split assignments
    
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

        # Append the parsed information to the mappings list
        mappings.append({
            'ProbeID': probe_id,
            'Ref_DB': ref_db,
            'Location': location,
            'GeneSymbol': gene_symbol,
            'ChromosomalLocation': chromosomal_locations,
            'EntrezGeneID': entrez_gene_id
        })

# Convert the list of mappings to a DataFrame
mapping_df = pd.DataFrame(mappings)

# Removing entries without a GeneSymbol
mapping_df = mapping_df[mapping_df['GeneSymbol'] != '']

# Combine expression data for all samples into a single DataFrame
expression_data = pd.DataFrame()


# Extract expression data for each sample
for gsm_name, gsm in gse.gsms.items():
    # Convert the table to a DataFrame and set 'ID_REF' as the index
    df = pd.DataFrame(gsm.table).set_index('ID_REF')
    # Select the VALUE column (assuming expression values are in 'VALUE'; adjust if necessary)
    expression_data[gsm_name] = df['VALUE']

# Reset index to turn 'ID_REF' into a column, facilitating the merge
expression_data.reset_index(inplace=True)

id_ref_set = set(expression_data['ID_REF'])
probe_id_set = set(mapping_df['ProbeID'])

# Find the intersection (common elements) between the two sets
common_ids = id_ref_set.intersection(probe_id_set)

# Convert the set of common IDs back to a list if you need to use it for indexing or further processing
common_ids_list = list(common_ids)

print(f"Number of common IDs: {len(common_ids)}")
# If you want to see the common IDs, you can print common_ids_list or a portion of it
print(common_ids_list[:10])  # Example: Print the first 10 common IDs


# Merge the expression data with the gene mapping data
# Note: This will align the expression data with the mapping based on ProbeID
merged_data = pd.merge(expression_data, mapping_df, left_on='ID_REF', right_on='ProbeID', how='left')

# Optionally, save the merged data to a CSV file
merged_data.to_csv('GSE44077_merged_expression_gene_mapping.csv', index=False)

print('Merged expression data and gene mapping saved to GSE44077_merged_expression_gene_mapping.csv.')

# Step 4: Save to CSV
mapping_df.to_csv('GSE44077_gene_mapping.csv', index=False)

print('Mapping saved to GSE44077_gene_mapping.csv.')

