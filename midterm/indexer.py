import os
import re
import time
import zstandard as zstd
from bs4 import BeautifulSoup
from tqdm import tqdm


class CollectionStats:

    def __init__(self, index_path):
        self.D = 0  # Colection size in terms of documents
        self.N = 0  # Colection size in terms of tokens
        self.C = 0  # Colection size in terms of characters

        self.indexFile = os.path.join(index_path, "colstats.txt")

    def numberOfDocuments(self):
        return self.D

    # Size of the collection
    def numberOfTokens(self):
        return self.N

    # Avg. # of tokens per document
    def averageTokensPerDoc(self):
        return 0 if self.D == 0 else self.N / self.D

    # Avg. # of chars per token
    def averageCharsPerToken(self):
        return 0 if self.N == 0 else self.C / self.N

    # Collection Stats writer function
    def write(self):
        if os.path.isfile(self.indexFile):
            os.remove(self.self.indexFile)
        with open(self.indexFile, 'w') as indexfile:
            indexfile.write(f"{self.N},{self.D},{self.C}")

    # Collection Stats loader function
    def load(self):
        with open(self.indexFile, 'r') as indexfile:
            self.N, self.D, self.C = [
                int(strval) for strval in indexfile.readline().split(",")]


class Lexicon:

    def __init__(self, index_path):
        # Implement this method
        self.lexicon = {}  # term:[idx, df, TF] mappings
        self.idx2term = []  # idx:term
        self.indexfile_lexicon = os.path.join(index_path, "lexicon.txt")
        self.indexfile_idx2term = os.path.join(index_path, "idx2term.txt")

    # add the token to the dictionary
    def add(self, token):
        try:
            idx, df, TF = self.lexicon.get(token)
            self.lexicon[token][2] += 1  # increment TF
        except:
            idx = len(self.idx2term)
            self.idx2term.append(token)
            self.lexicon[token] = [idx, 0, 1]

        return self.lexicon[token]  # return [idx, df, TF]

    # returns the number of terms in the dictionary
    def __len__(self):
        return len(self.lexicon)

    # Accumulates the "df" to the existing value of "df"
    def updateDF(self, termidx, df):
        term = self.idx2term[termidx]
        idx, _df, _TF = self.lexicon.get(term)
        if idx != termidx:
            raise ValueError(
                f"For term {term}, idx in lexicon and termidx are mismatch: {idx} and {termidx}")

        self.lexicon[term][1] = _df + df
        return self.lexicon[term]

    # Returns the index of the term in the lexion
    def getIdx(self, term):
        return -1 if not self.lexicon[term] else self.lexicon[term][0]

    # Document Frequency of the input term
    def getDF(self, term):
        return 0 if not self.lexicon.get(term) else self.lexicon[term][1]

    # Total Frequency of the input term
    def getTF(self, term):
        return 0 if not self.lexicon.get(term) else self.lexicon[term][2]

    # Returns the term for the given index idx
    def getTerm(self, idx):
        return "" if idx < 0 or idx > len(self.idx2term) else self.idx2term[idx]

    # Returns total DF over all terms
    def getTotalDF(self):
        return sum([values[1] for values in self.lexicon.values()])

    # Returns total TF over all terms, i.e. Collection Size N
    def getTotalTF(self):
        return sum([values[2] for values in self.lexicon.values()])

    # Lexicon writer function
    def write(self):
        # lexicon
        if os.path.isfile(self.indexfile_lexicon):
            os.remove(self.indexfile_lexicon)
        with open(self.indexfile_lexicon, 'w') as indexfile:
            for item in self.lexicon.items():
                key = item[0]
                values = ",".join([str(value) for value in item[1]])
                indexfile.write(f"{key}->{values}\n")

        # idx2docid
        if os.path.isfile(self.indexfile_idx2term):
            os.remove(self.indexfile_idx2term)
        with open(self.indexfile_idx2term, 'w') as indexfile:
            for term in self.idx2term:
                indexfile.write(f"{term}\n")

    # Lexicon loader function
    def load(self):
        self.lexicon = {}
        with open(self.indexfile_lexicon, 'r') as indexfile:
            for line in indexfile.readlines():
                key, values_ = line.split("->")
                values = [int(value) for value in values_.split(",")]
                self.lexicon[key] = values

        # idx2term
        self.idx2term = []
        with open(self.indexfile_idx2term, 'r') as indexfile:
            for line in indexfile.readlines():
              self.idx2term.append(line.replace("\n",""))


class Documents:

    def __init__(self, index_path):
        # Implement this method
        self.docList = {}  # docID:(idx,length)
        self.idx2docid = []  # idx:docID
        self.indexFile_docList = os.path.join(index_path, 'docList.txt')
        self.indexFile_idx2docid = os.path.join(index_path, 'idx2docid.txt')

    def __len__(self):
        return len(self.docList)

    def add(self, docID, length=0):
        idx = len(self.idx2docid)
        self.idx2docid.append(docID)
        self.docList[docID] = (idx, length)
        return (idx, length)

    # Number of the documents
    def __len__(self):
        return len(self.docList)

    # Returns the index of the document
    def getLength(self, docID):
        return 0 if not self.docList.get(docID) else self.docList[docID][1]

    # Returns the index of the document
    def getIdx(self, docID):
        return self.docList.get(docID)

    # Returns the name of document for the given index idx
    def getDocID(self, idx):
        return self.idx2docid[idx]

    # Documents writer function
    def write(self):
        # docList
        if os.path.isfile(self.indexFile_docList):
            os.remove(self.indexFile_docList)
        with open(self.indexFile_docList, 'w') as indexfile:
            for item in self.docList.items():
                key = item[0]
                values = ",".join([str(value) for value in item[1]])
                indexfile.write(f"{key}->{values}\n")

        # idx2docid
        if os.path.isfile(self.indexFile_idx2docid):
            os.remove(self.indexFile_idx2docid)
        with open(self.indexFile_idx2docid, 'w') as indexfile:
            for docid in self.idx2docid:
                indexfile.write(f"{docid}\n")

    # Documents loader function
    def load(self):
        # docList
        self.docList = {}
        with open(self.indexFile_docList, 'r') as indexfile:
            for line in indexfile.readlines():
                key, values_ = line.split("->")
                values = [int(value) for value in values_.split(",")]
                self.docList[key] = values

        # idx2docid
        self.idx2docid = []
        with open(self.indexFile_idx2docid, 'r') as indexfile:
            for line in indexfile.readlines():
                self.idx2docid.append(line.replace("\n",""))


class Index:

    def __init__(self, index_path):
        self.index_path = index_path
        self.invertedIndex = os.path.join(index_path, "inverted.txt")

        self.lexicon = Lexicon(index_path)
        self.documents = Documents(index_path)
        self.collectionStats = CollectionStats(index_path)

        # in memory postings lists
        self.postingsList = {}

    # ------------------------------------------------------------------
    # ------------------------ BSBI ------------------------------------
    # ------------------------------------------------------------------
    def list_files(self, dir_path):
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_file():
                    yield os.path.join(dir_path, entry.name)

    def documentReader(self, collection_path):
        dctx = zstd.ZstdDecompressor()
        for zstd_filename in self.list_files(collection_path):
            with open(zstd_filename, 'rb') as zstd_file:
                decompressed_data = dctx.decompress(zstd_file.read())
                file_content = decompressed_data.decode('utf-8')
                soup = BeautifulSoup(file_content, 'html.parser')
                docs = soup.find_all('doc')
                for doc in docs:
                    docID = doc.find('docid').text
                    text = doc.find('text').text.strip()
                    yield docID, text

    def writeBlockToDisk(self, n, current_block):
        blockFilePath = os.path.join(self.index_path, f"block_{n}.txt")

        if os.path.isfile(blockFilePath):
            os.remove(blockFilePath)

        with open(blockFilePath, 'w') as blockfile:
            # sort the keys
            # use Collator object's getSortKey() function as call-back for sorted()
            sorted_items = sorted(current_block.items())
            for termidx, postings in sorted_items:
                self.lexicon.updateDF(termidx, len(postings))
                posting_str = ";".join(
                    [f"{posting[0]}:{posting[1]}" for posting in postings])
                blockfile.write(f"{termidx}->{posting_str}\n")

    # WARNING: block_file1's block number must be less than that of block_file2
    def merge_sorted_blocks(self, block_file1, block_file2, output_file):
        if os.path.isfile(output_file):
            print(f"Output file for merging exists, deleting {output_file}")
            os.remove(output_file)

        # Open the input block files and the output block file
        with open(block_file1, 'r') as file1, open(block_file2, 'r') as file2, open(output_file, 'w') as output:
            # Read the first line from each input block file
            line1 = file1.readline().strip()
            line2 = file2.readline().strip()

            # Iterate through the input block files until we reach the end of one of them
            while line1 and line2:
                # Compare the current terms and write the smaller one to the output file
                termidx1, postings1 = line1.split("->")
                termidx2, postings2 = line2.split("->")
                idx1 = int(termidx1)
                idx2 = int(termidx2)
                if idx1 < idx2:
                    output.write(line1 + '\n')
                    line1 = file1.readline().strip()
                elif idx1 > idx2:
                    output.write(line2 + '\n')
                    line2 = file2.readline().strip()
                else:  # terms are the same, so merge postings
                    merged_line = termidx1 + "->" + postings1 + ";" + postings2
                    output.write(merged_line + '\n')
                    line1 = file1.readline().strip()
                    line2 = file2.readline().strip()

            # Write any remaining lines from file 1 to the output file
            while line1:
                output.write(line1 + '\n')
                line1 = file1.readline().strip()

            # Write any remaining lines from file 2 to the output file
            while line2:
                output.write(line2 + '\n')
                line2 = file2.readline().strip()

    # -------------------------------------------------------------------------
    def create(self, collection_path):
        # Clear previous blocks & indices if exists
        with os.scandir(self.index_path) as entries:
            for entry in entries:
                if entry.is_file():
                    os.remove(entry.path)

        # Create Inverted Index using BSB Indexing
        block_size = 10000
        block_no = 0
        current_block = {}  # term:[(docNo,tf),(docNo,tf),...]

        for docID, text in tqdm(self.documentReader(collection_path)):

            tokenized_doc = [token.lower()
                             for token in re.findall(r'\b\w+\b', text)]

            docidx, _ = self.documents.add(docID, len(tokenized_doc))

            # Update collection stats
            self.collectionStats.D += 1
            self.collectionStats.N += len(tokenized_doc)

            # invert document into current_block
            for token in tokenized_doc:
                self.collectionStats.C += len(token)
                termidx, _, _ = self.lexicon.add(token)
                try:
                    currdocidx, tf = current_block.get(termidx)[-1]
                    if currdocidx == docidx:
                        current_block[termidx][-1] = (docidx, tf + 1)
                    else:
                        current_block[termidx].append((docidx, 1))
                except:
                    current_block.setdefault(termidx, []).append((docidx, 1))

            if self.collectionStats.D % block_size == 0:
                self.writeBlockToDisk(block_no, current_block)
                # Reset current block & increment block count n
                current_block = {}
                block_no += 1

        # Write if there are trailing docs in the current_block
        if len(current_block) > 0:
            self.writeBlockToDisk(block_no, current_block)
            current_block = None

        # List all of the block files in the index directory
        file_list = self.list_files(self.index_path)

        blocks = []
        for file in file_list:
            filename, extension = os.path.splitext(file)
            block_number = filename.split("/")[-1].split('_')[1]
            blocks.append((block_number, file))

        # Blocks must be sorted by block_number
        blocks.sort(key=lambda pair: int(pair[0]))

        queue = []
        queue_2 = []

        for block in blocks:
            queue.append(block)

        start_time = time.time()

        while len(queue) > 1:
            block_1_file = queue.pop(0)[1]
            block_2_file = queue.pop(0)[1]

            filename1, _ = os.path.splitext(block_1_file)
            filename2, _ = os.path.splitext(block_2_file)

            output_block_num = filename1.split(
                '/')[-1].split('_')[-1] + filename2.split('/')[-1].split('_')[-1]
            output_block_file = os.path.join(
                self.index_path, f"block_{output_block_num}.txt")

            blocks.append(output_block_file)

            print(
                f"Blocks {filename1.split('/')[-1]} and {filename2.split('/')[-1]} are in progress...")

            self.merge_sorted_blocks(
                block_1_file, block_2_file, output_block_file)
            os.remove(block_1_file)
            os.remove(block_2_file)

            print(
                f"Merging is done into -> {output_block_file.split('/')[-1]}")
            elapsed = time.time() - start_time
            print(f"Elapsed time: {round(elapsed/60)}:{round(elapsed % 60)}\n")

            queue_2.append((output_block_num, output_block_file))

            if (len(queue) < 2) & (len(queue_2) > 0):
                while len(queue) > 0:
                    queue_2.append(queue.pop(0))

                while len(queue_2) > 0:
                    queue.append(queue_2.pop(0))

        # Resulting merged block file name
        merged_block_file = queue.pop(0)[1]

        print(f"\nAll blocks are merged into: {merged_block_file}")
        elapsed = time.time() - start_time
        print(f"Elapsed time: {round(elapsed/60)}:{round(elapsed % 60)}\n")

        # Rename the merged-block as inverted.txt
        os.rename(merged_block_file, self.invertedIndex)
        print(f"\nMerged block file is renamed as: {self.invertedIndex}")

        # Write helper classes
        self.collectionStats.write()
        self.lexicon.write()
        self.documents.write()

        # Load indices
        self.load()

    # Loads the inverted index from self.index_path
    # ---------------------------------------------------------------
    def load(self):
        # Implement the loading of the inverted index from index_path
        print("\nLoading Inverted Index...")
        with open(self.invertedIndex, 'r') as index_file:
            for line in tqdm(index_file.readlines()):
                key_, values_ = line.split("->")
                values = [(int(docidx), int(tf)) for docidx, tf in [
                    value.split(":") for value in values_.split(";")]]
                self.postingsList[int(key_)] = values

        # After the inverted index is loaded
        print("\nLoading Collection Stats...")
        self.collectionStats.load()
        print("\nLoading Lexicon...")
        self.lexicon.load()
        print("\nLoading Document List...")
        self.documents.load()
