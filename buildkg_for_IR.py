# buildkg_for_IR.py
import pandas as pd
import spacy
import re
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
    def __init__(self, host, port, username, password, task_name, csv_path):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.task_name = task_name
        self.space_name = f"medgraph_{task_name}"  # Dynamic graph space name
        self.csv_path = csv_path
        self.connection_pool = None
        self.nlp_model = None  # Placeholder for the NLP model
        logging.info(f"Initialized builder for task '{self.task_name}' using space '{self.space_name}'")

    def connect(self):
        """Initializes the NebulaGraph connection pool."""
        logging.info(f"Connecting to NebulaGraph at {self.host}:{self.port}...")
        try:
            config = Config()
            config.max_connection_pool_size = 10
            self.connection_pool = ConnectionPool()
            self.connection_pool.init([(self.host, self.port)], config)
            # Verify connection
            with self.get_session() as session:
                result = session.execute("SHOW HOSTS")
                if result.is_succeeded():
                    logging.info("Connection successful.")
                else:
                    raise ConnectionError(f"Failed to connect: {result.error_msg()}")
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

    def sanitize_text(self, text):
        return str(text).replace("\\", "\\\\")\
                        .replace('"', '\\"')\
                        .replace("\n", " ")\
                        .replace("\r", " ")

    def sanitize_url_doi(self, text):
        return str(text).replace("\n", "").replace("\r", "").strip()

    def _create_schema(self):
        """Creates the graph space and schema if they don't exist."""
        with self.get_session() as session:
            logging.info(f"Ensuring graph space '{self.space_name}' exists...")
            session.execute(f"CREATE SPACE IF NOT EXISTS {self.space_name}(vid_type=FIXED_STRING(256));")
            time.sleep(5)  # Allow time for space creation to propagate in the cluster
            session.execute(f"USE {self.space_name};")
            
            logging.info("Defining graph schema (Tags and Edges)...")
            # A unified, comprehensive schema for the Paper tag
            session.execute("""
                CREATE TAG IF NOT EXISTS Paper(
                    pmid string, title string, doi string, 
                    abstract string, url string, is_paper_exists string
                );
            """)
            session.execute("CREATE TAG INDEX IF NOT EXISTS paper_index ON Paper(pmid(256));") # Index for faster lookups
            session.execute("CREATE TAG IF NOT EXISTS Concept(concept_id string, name string, label string);")
            session.execute("CREATE EDGE IF NOT EXISTS MENTIONS();")
            time.sleep(5) # Allow time for schema changes to apply
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
        
        logging.info(f"Inserting {len(df)} papers into '{self.space_name}'...")
        
        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")
            for idx, row in df.iterrows():
                pmid = self.sanitize_text(row["PMID"])
                title = self.sanitize_text(row["Title"])
                doi = self.sanitize_url_doi(row["DOI"])
                abstract = self.sanitize_text(row["Abstract"])
                url = self.sanitize_url_doi(row["url"])
                is_paper_exists = str(row["is_paper_exists"]).lower()

                # Use PMID as Vertex ID (must be a string)
                vertex_id = f'"{pmid}"'

                insert_query = f'''
                INSERT VERTEX Paper(pmid, title, doi, abstract, url, is_paper_exists)
                VALUES {vertex_id}: ("{pmid}", "{title}", "{doi}", "{abstract}", "{url}", "{is_paper_exists}");
                '''

                try:
                    result = session.execute(insert_query)
                    if result.is_succeeded():
                        print(f"[✓] Inserted: {pmid}")
                    else:
                        print(f"[x] Failed: {pmid} | Error: {result.error_msg()}")
                except Exception as e:
                    print(f"[!] Exception on {pmid}: {e}")

    def _load_nlp_model(self):
        """Loads the SciSpaCy model once, only when needed."""
        if self.nlp_model is None:
            logging.info("Loading SciSpaCy model 'en_ner_bionlp13cg_md'...")
            try:
                self.nlp_model = spacy.load("en_ner_bionlp13cg_md")
                logging.info("SciSpaCy model loaded successfully.")
            except OSError:
                logging.error("Could not find SciSpaCy model 'en_ner_bionlp13cg_md'.")
                logging.error("Please run: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bionlp13cg_md-0.5.1.tar.gz")
                raise

    def insert_concept(self, session, paper_id, concept_name, label):
        """Inserts a concept vertex and an edge linking it to a paper."""
        # Sanitize concept name for use in queries and as an ID
        sanitized_name = self.sanitize_text(concept_name)
        concept_id = sanitized_name.lower().replace(" ", "_").replace('"', '')

        # Use quotes for string literals in the query
        paper_vid = f'"{paper_id}"'
        concept_vid = f'"{concept_id}"'

        query = f'''
        INSERT VERTEX IF NOT EXISTS Concept(concept_id, name, label)
            VALUES {concept_vid}:("{concept_id}", "{sanitized_name}", "{label}");
        INSERT EDGE IF NOT EXISTS MENTIONS()
            VALUES {paper_vid}->{concept_vid}:();
        '''
        result = session.execute(query)
        if not result.is_succeeded():
            logging.warning(f"Failed to insert concept '{sanitized_name}' for paper {paper_id}: {result.error_msg()}")

    def process_abstract(self, abstract: str, paper_id: str, session):
        """Processes a single abstract to find and insert concepts using the pre-loaded NLP model."""
        doc = self.nlp_model(abstract)
        for ent in doc.ents:
            concept_name = ent.text.strip()
            if concept_name: # Ensure we don't insert empty concepts
                self.insert_concept(session, paper_id, concept_name, ent.label_)

    def process_all_papers(self):
        """Fetches all papers from the graph and processes their abstracts for concept extraction."""
        self._load_nlp_model()  # Ensure the NLP model is loaded before starting
        
        with self.get_session() as session:
            session.execute(f"USE {self.space_name};")
            
            logging.info("Fetching all papers to process for concept extraction...")
            # Query all Paper nodes (pmid and abstract fields)
            result = session.execute("MATCH (p:Paper) RETURN p;")

            if not result.is_succeeded():
                print("Query failed:", result.error_msg())
                return

            for row in result:
                # Access the first element from .values()
                node = row.values()[0]
                
                # Convert the node to a string representation
                node_str = str(node)
                
                # Extract key-value pairs using regex
                attributes = re.findall(r"(\w+): \"([^\"]*)\"", node_str)
                
                vertex_id = attributes[3][1]     # internal Nebula vertex ID
                pmid = attributes[3][1]           # paper id
                abstract = attributes[0][1]       # abstract text

                if abstract.strip():
                    try:
                        self.process_abstract(abstract, vertex_id, session)
                    except Exception as e:
                        print(f"Failed to process paper {pmid}: {e}")

    def run(self):
        """Executes the full knowledge graph construction pipeline for the given task."""
        try:
            self.connect()
            self._create_schema()
            self.insert_papers()
            self.process_all_papers()
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
    
    args = parser.parse_args()

    builder = KnowledgeGraphBuilder(
        host=args.db_host,
        port=args.db_port,
        username=args.db_user,
        password=args.db_password,
        task_name=args.task,
        csv_path=args.csv_path
    )
    
    builder.run()

if __name__ == "__main__":
    main()