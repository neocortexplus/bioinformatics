from data_handler import GEODataManager,DisplayOptionsManager,GEODataParser,GEODataMerger
from data_manipulator import DataManipulator  # Make sure this is the correct import path
from fs_methods import FeatureSelector



if __name__ == "__main__":



    DisplayOptionsManager.toggle_pandas_display_options(False)

    # gse_id = "GSE44077"
    gse_id = "GSE19804"
    manager  = GEODataManager(gse_id=gse_id)
    gse = manager.download_dataset()
    manager.display_metadata()
    parser = GEODataParser(gse=gse)
    mapping_df = parser.parse_gpl_data()
    expression_data = parser.compile_expression_data()
    merger = GEODataMerger(expression_data=expression_data, mapping_df=mapping_df)
    merged_data = merger.merge_data()

    cleaned_data = manager.clean_entrez_gene_id(merged_data=merged_data, column_name='EntrezGeneID')

    cleaned_data = manager.modify_dataframe(cleaned_data,None,['ProbeID',"ID_REF",
        'Ref_DB', 'Location', 'GeneSymbol', 'ChromosomalLocation',"EntrezGeneID"],"EntrezGeneID")

    final_data = parser.add_labels_to_expression_data(cleaned_data , metadata_key='characteristics_ch1', label_prefix='tissue', separator=':')


    manager.save_to_csv(dataframe=final_data, merged_file= f"{gse_id}_final_data.csv")

    #verify the dataset

    gse_id = "GSE44077"
    manager  = GEODataManager(gse_id=gse_id)
    df = manager.load_csv('final_data.csv')

    DataManipulator = DataManipulator(df)
    df_shuffled = DataManipulator.shuffle_rows()
    df_noised = DataManipulator.add_random_noise('A', noise_level=0.05)
    df_randomized_col = DataManipulator.randomize_column('B')
    df_swapped = DataManipulator.swap_columns('A', 'B')
    df_synthetic = DataManipulator.add_synthetic_feature(lambda row, scale: row['A'] * scale + row['B'], 'D', scale=2)

    print(df.head())


    selector = FeatureSelector(dataframe=df, label_column='Label')
    selector.encode_labels()
    selector.split_data()
    
    # Train XGBoost and print top 10 features
    selector.train_xgboost()
    print("Top 10 Features from XGBoost:")
    print(selector.get_top_n_features(n=10))
    
    # Train Random Forest and print top 10 features
    selector.train_random_forest()
    print("\nTop 10 Features from Random Forest:")
    print(selector.get_top_n_features(n=10))
    
    # Apply t-test and print top 10 features
    print("\nTop 10 Features from t-test:")
    print(selector.apply_ttest(k=10))
