import pandas as pd
import spacy
import argparse
import logging
import time
from tqdm import tqdm
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from contextlib import contextmanager

# Configure logging to provide clear, informative output
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class KnowledgeGraphBuilder:
    """
    A generalized class to build a knowledge graph in NebulaGraph for various
    information retrieval tasks. It handles:
    1. Dynamic creation of graph spaces based on the task name.
    2. Creating a unified graph schema (Tags, Edges).
    3. Inserting paper data from different CSV structures in efficient batches.
    4. Processing abstracts with SciSpaCy to extract and link biomedical concepts.
    """
    def __init__(self, host, port, username, password, task_name, csv_path, batch_size=200):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.task_name = task_name
        self.space_name = f"medgraph_{task_name}"  # Dynamic graph space name
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.connection_pool = None
        self.nlp_model = None
        logging.info(f"Initialized builder for task '{self.task_name}' using space '{self.space_name}'")

    def connect(self):
        """Initializes the NebulaGraph connection pool."""
        logging.info(f"Connecting to NebulaGraph at {self.host}:{self.port}...")
        try:
            config = Config()
            config.max_connection_pool_size = 10
            self.connection_pool = ConnectionPool()
            self.connection_pool.init([(self.host, self.port)], config)
            logging.info("Connection successful.")
        except Exception as e:
            logging.error(f"Failed to connect to NebulaGraph: {e}")
            raise

    def close(self):
        """Closes the NebulaGraph connection pool."""
        if self.connection_pool:
            self.connection_pool.close()
            logging.info("Connection to NebulaGraph closed.")

    @contextmanager
    def get_session(self):
        """Provides a managed session from the connection pool."""
        session = None
        try:
            session = self.connection_pool.get_session(self.username, self.password)
            yield session
        except Exception as e:
            logging.error(f"Failed to get a session: {e}")
            raise
        finally:
            if session:
                session.release()

    @staticmethod
    def _sanitize(text, is_id=False):
        """Sanitizes text for Cypher queries."""
        if not isinstance(text, str):
            text = str(text)
        sanitized = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")
        if is_id:
            sanitized = sanitized.strip()
        return sanitized

    def _create_schema(self):
        """Creates the graph space and schema if they don't exist."""
        with self.get_session() as session:
            logging.info(f"Ensuring graph space '{self.space_name}' exists...")
            session.execute(f"CREATE SPACE IF NOT EXISTS {self.space_name}(vid_type=FIXED_STRING(256));")
            time.sleep(5) # Give time for space creation to propagate
            session.execute(f"USE {self.space_name};")
            
            logging.info("Defining graph schema (Tags and Edges)...")
            # A unified, comprehensive schema for the Paper tag
            session.execute("""
                CREATE TAG IF NOT EXISTS Paper(
                    pmid string, title string, doi string, 
                    abstract string, url string, is_paper_exists string
                );
            """)
            session.execute("CREATE TAG IF NOT EXISTS Concept(concept_id string, name string, label string);")
            session.execute("CREATE EDGE IF NOT EXISTS MENTIONS();")
            time.sleep(5)
            logging.info("Schema is ready.")

    def insert_papers(self):
        """Loads data from CSV and inserts Paper nodes in batches."""
        logging.info(f"Loading data from {self.csv_path}")
        try:
            df = pd.read_csv(self.csv_path)
            df.fillna("", inplace=True)
        except FileNotFoundError:
            logging.error(f"CSV file not found at: {self.csv_path}")
            return
        
        logging.info(f"Inserting {len(df)} papers into '{self.space_name}' in batches of {self.batch_size}...")
        
        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")
            for i in tqdm(range(0, len(df), self.batch_size), desc="Inserting Papers"):
                batch_df = df.iloc[i:i + self.batch_size]
                
                values = []
                for _, row in batch_df.iterrows():
                    # Safely get data from columns that may or may not exist in the CSV
                    pmid = self._sanitize(row.get("PMID", ""), is_id=True)
                    if not pmid: continue # PMID is essential for the Vertex ID

                    title = self._sanitize(row.get("Title", ""))
                    doi = self._sanitize(row.get("DOI", ""))
                    abstract = self._sanitize(row.get("Abstract", ""))
                    url = self._sanitize(row.get("url", ""))
                    is_exists = self._sanitize(str(row.get("is_paper_exists", "")).lower())

                    values.append(
                        f'"{pmid}":("{pmid}", "{title}", "{doi}", "{abstract}", "{url}", "{is_exists}")'
                    )
                
                if not values: continue

                insert_query = f"""
                INSERT VERTEX Paper(pmid, title, doi, abstract, url, is_paper_exists)
                VALUES {', '.join(values)};
                """
                
                result = session.execute(insert_query)
                if not result.is_succeeded():
                    logging.error(f"Failed to insert batch at index {i}: {result.error_msg()}")

    def extract_and_link_concepts(self):
        """Fetches papers with abstracts, extracts concepts, and links them."""
        logging.info("Loading SciSpaCy model 'en_ner_bionlp13cg_md'...")
        try:
            self.nlp_model = spacy.load("en_ner_bionlp13cg_md")
        except IOError:
            logging.error("SciSpaCy model not found. Please run the download command.")
            return

        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")
            
            logging.info("Fetching all paper abstracts from the graph...")
            fetch_query = "MATCH (p:Paper) RETURN p.pmid AS pmid, p.abstract AS abstract;"
            result = session.execute(fetch_query)
            
            if not result.is_succeeded():
                logging.error(f"Failed to fetch papers: {result.error_msg()}")
                return

            # Filter for papers that actually have an abstract to process
            papers = [(row['pmid'].as_string(), row['abstract'].as_string()) for row in result]
            papers_with_abstracts = [p for p in papers if p[1] and p[1].strip()]

            if not papers_with_abstracts:
                logging.warning("No papers with abstracts found in this graph. Skipping concept extraction.")
                return

            texts = [p[1] for p in papers_with_abstracts]
            pmids = [p[0] for p in papers_with_abstracts]

            logging.info(f"Processing {len(texts)} abstracts to find concepts...")
            concepts_to_insert = set()
            edges_to_insert = []
            
            for doc, pmid in tqdm(zip(self.nlp_model.pipe(texts, batch_size=50), pmids), total=len(texts), desc="Extracting Concepts"):
                for ent in doc.ents:
                    concept_name = self._sanitize(ent.text.strip())
                    if not concept_name: continue
                    concept_label = self._sanitize(ent.label_)
                    concept_id = concept_name.lower().replace(" ", "_")
                    
                    concepts_to_insert.add((concept_id, concept_name, concept_label))
                    edges_to_insert.append(f'"{pmid}"->"{concept_id}":()')

            logging.info(f"Found {len(concepts_to_insert)} unique concepts. Inserting them...")
            self._insert_batch("Concept", list(concepts_to_insert))

            logging.info(f"Found {len(edges_to_insert)} mentions. Inserting edges...")
            self._insert_batch("Edge", edges_to_insert)

    def _insert_batch(self, entity_type, data):
        """Generic method to insert vertices or edges in batches."""
        if not data: return

        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")
            desc = f"Inserting {entity_type}s"
            for i in tqdm(range(0, len(data), self.batch_size), desc=desc):
                batch = data[i:i + self.batch_size]
                if entity_type == "Concept":
                    values = [f'"{cid}":("{cid}", "{name}", "{label}")' for cid, name, label in batch]
                    query = f"INSERT VERTEX IF NOT EXISTS Concept(concept_id, name, label) VALUES {', '.join(values)};"
                elif entity_type == "Edge":
                    query = f"INSERT EDGE IF NOT EXISTS MENTIONS() VALUES {', '.join(batch)};"
                else:
                    return

                result = session.execute(query)
                if not result.is_succeeded():
                    logging.error(f"Failed to insert {entity_type} batch: {result.error_msg()}")

    def run(self):
        """Executes the full knowledge graph construction pipeline for the given task."""
        try:
            self.connect()
            self._create_schema()
            self.insert_papers()
            self.extract_and_link_concepts()
            logging.info(f"Knowledge graph build for task '{self.task_name}' is complete!")
        except Exception as e:
            logging.error(f"A critical error occurred during the build process: {e}")
        finally:
            self.close()

def main():
    parser = argparse.ArgumentParser(description="Build a NebulaGraph Knowledge Graph for specific IR tasks.")
    
    # Task-specific arguments
    parser.add_argument("--task", type=str, required=True, 
                        choices=['title2url', 'abstract2url', 'pmid2title', 'url2title'], 
                        help="The specific IR task to build the graph for. This determines the graph space name.")
    parser.add_argument("--csv_path", type=str, required=True, 
                        help="Path to the input CSV file for the specified task.")
    
    # NebulaGraph connection arguments
    parser.add_argument("--db_host", type=str, default="127.0.0.1", help="NebulaGraph host address.")
    parser.add_argument("--db_port", type=int, default=9669, help="NebulaGraph port.")
    parser.add_argument("--db_user", type=str, default="root", help="NebulaGraph username.")
    parser.add_argument("--db_password", type=str, default="nebula", help="NebulaGraph password.")
    parser.add_argument("--batch_size", type=int, default=200, help="Number of records to insert per batch.")
    
    args = parser.parse_args()

    builder = KnowledgeGraphBuilder(
        host=args.db_host,
        port=args.db_port,
        username=args.db_user,
        password=args.db_password,
        task_name=args.task,
        csv_path=args.csv_path,
        batch_size=args.batch_size
    )
    
    builder.run()

if __name__ == "__main__":
    main()