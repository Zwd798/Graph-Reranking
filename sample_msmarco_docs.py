import pandas as pd

def get_random_data(sampled_msmarco_docs, sampled_docids):
    sampled_msmarco = pd.read_csv('msmarco_doc_reranking/msmarco-docs.tsv',
                            sep="\t",header=None, names=['docid', 'url', 'title', 'body'], nrows=500000)
    random_samples = (
        sampled_msmarco[~sampled_msmarco['docid'].isin(sampled_docids)]
        .drop_duplicates(subset='docid')   # ensure uniqueness
        .iloc[:len(sampled_msmarco_docs)]  # match size
    )

    return random_samples


def generate_sampled_msmarco_dataset(n_qid=1000):
    msmarco_doctrain_top100 = pd.read_csv('/home/nxz190009/phd/graph_reranking/msmarco_doc_reranking/msmarco-doctrain-top100', 
                                        sep="\s+",header=None, 
                                        names=["qid", "placeholder1", "docid", "placeholder2", "placeholder3", "placeholder4"])

    sampled_qids = msmarco_doctrain_top100['qid'].sample(n=n_qid, random_state=42).tolist()
    msmarco_doctrain_top100_sampled = msmarco_doctrain_top100[msmarco_doctrain_top100['qid'].isin(sampled_qids)]
    sampled_docids = msmarco_doctrain_top100_sampled['docid'].tolist()

    chunksize = 10**6 
    filtered_rows = []

    for chunk in pd.read_csv('msmarco_doc_reranking/msmarco-docs.tsv',
                                sep="\t",header=None, names=['docid', 'url', 'title', 'body'], chunksize=chunksize):
        filtered = chunk[chunk['docid'].isin(sampled_docids)]
        filtered_rows.append(filtered)

    # Concatenate all filtered rows if needed
    sampled_msmarco_docs = pd.concat(filtered_rows, ignore_index=True)

    random_samples = get_random_data(sampled_msmarco_docs, sampled_docids)
    
    sampled_msmarco_docs = pd.concat([sampled_msmarco_docs,random_samples])

    folder_path = 'my_folder'
    os.makedirs(folder_path, exist_ok=True)
    
    sampled_msmarco_docs.to_csv(f"data/sampled_msmarco-docs.csv", index=False)

if __name__=="__main__":
    generate_sampled_msmarco_dataset()