from datetime import datetime

from nerd.data.util.mc_generator import McGenerator
from nerd.data.util.readers.BBNDataReader import BBNDataReader
from nerd.data.util.readers.WNUTDataReader import WNUTDataReader

from nerd.data.util.readers.JSONDataReader import JSONDataReader
from nerd.memory_retrieval.mention_driven_retrieval import MentionDrivenRetrieval

if __name__ == '__main__':

    # SETUP CONFIG
    # data_set = 'WNUT'
    # data_file_path = '/home/nikhil/workspace/data61/local/mwe_ner/ner/data/WNUT/train'
    t = datetime.now()
    data_set = 'JSON'
    data_file_path = '/home/nikhil/Downloads/BBN/BBN/test.json'

    # Note use model_path = None to save memory and not load text_embeddings if not required
    # model_path = '/home/nikhil/workspace/data61/local/mwe_ner/ner/pretrained_models/word2vec/GoogleNews-vectors-negative300.bin'
    # model_path = '/home/nikhil/workspace/data61/local/wordvec/skipgram_v200_w5'
    model_path = None
    model_is_binary = False

    # Memory related data
    db_path = '/home/nikhil/workspace/data61/local/db/skip_contxtEmb1000.db'
    index_path = '/home/nikhil/workspace/data61/local/db/skip_contxtEmb1000.idx'
    m4mat_emb_path = '/home/nikhil/workspace/data61/local/emb'

    # gpu_1 settings
    # db_path = '/media/data3tb2/yibing/nosqldb/database/skip_contxtEmb.db'
    # index_path = '/media/data3tb2/yibing/nosqldb/database/skip_contxtEmb_lsh_index.idx'
    # m4mat_emb_path = '/media/data3tb2/yibing/nosqldb/skip_mat'
    #
    # model_path = '/media/data3tb2/yibing/nosqldb/skipgram_v200_w5'
    # Note use model_path = None to save memory and not load text_embeddings if not required
    # model_path = '/home/nikhil/workspace/data61/local/mwe_ner/ner/pretrained_models/word2vec/GoogleNews-vectors-negative300.bin'

    reader = WNUTDataReader

    if data_set == 'BBN':
        reader = BBNDataReader
    if data_set == 'JSON':
        reader = JSONDataReader

    # END OF SETUP CONFIG
    # ----------------------------------------------- #

    # Read data
    st = datetime.now()
    data = reader.readFile(data_file_path, modelPath=model_path, model_is_binary=model_is_binary)
    dt = datetime.now() - st
    print('Compiled data in %d seconds...' % (dt.total_seconds()))

    # Generate MC

    st = datetime.now()
    mcGen = McGenerator(data)

    st = datetime.now()

    mc = mcGen.get_positive_candidate()

    mc_negs = mcGen.get_negative_candidates_from_sentence(mc.sentence_index)
    print(data[mc.sentence_index].text)
    print(data[mc.sentence_index].labels)
    print('Positive Candidate ........')
    print(mc)
    print('Negative Candidates .......')
    print(mc_negs)

    # dt = datetime.now() - st
    # print('Generated MC in %d seconds...' % (dt.total_seconds()))
    # print(mc)
    #
    #
    # MentionDrivenRetrieval.init_memory(db_path=db_path, index_file_path=index_path, emb_context_dir=m4mat_emb_path)
    # st = datetime.now()
    # MentionDrivenRetrieval.retrieve_mentions(mc)
    # dt = datetime.now() - st
    # print('Retrieved candidate  in %d seconds...' % (dt.total_seconds()))
    #
    # t = datetime.now() - t
    #
    # print('Process completed in %d seconds...' % (t.total_seconds()))