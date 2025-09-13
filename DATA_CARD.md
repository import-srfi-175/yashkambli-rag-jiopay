# Data Card for JioPay RAG Chatbot

## Dataset Overview
This dataset contains publicly accessible information from JioPay's business website and help center, collected for building a customer support RAG chatbot.

## Data Sources

### Primary Sources
1. **JioPay Business Website**
   - URL: https://jiopay.com/business
   - Content: Business information, features, pricing, integration details
   - Collection Method: Web scraping with multiple pipelines

2. **JioPay Help Center/FAQs**
   - URL: https://jiopay.com/help (or similar)
   - Content: Frequently asked questions, troubleshooting guides
   - Collection Method: Structured data extraction

### Additional Sources
- Any other publicly accessible JioPay documentation
- Public API documentation (if available)
- Community forums and public discussions

## Data Collection Details

### Scraping Pipelines Used
1. **requests + BeautifulSoup4**: Primary scraping method
2. **trafilatura**: Readability-focused extraction
3. **Playwright**: Dynamic content handling (if needed)

### Collection Metrics
- **Total Pages Scraped**: [To be filled after scraping]
- **Total Tokens**: [To be filled after processing]
- **Coverage**: Business pages, FAQ sections, help documentation
- **Noise Ratio**: [To be calculated during processing]
- **Throughput**: [To be measured during scraping]

### Data Quality
- **Cleanliness**: HTML boilerplate removed, structure preserved
- **Completeness**: All publicly accessible content included
- **Accuracy**: Source URLs maintained for verification

## Data Processing

### Chunking Strategies
1. **Fixed Chunking**: 256, 512, 1024 tokens with 0, 64, 128 overlap
2. **Semantic Chunking**: Sentence/paragraph boundary detection
3. **Structural Chunking**: HTML tag and heading-based segmentation
4. **Recursive Chunking**: Hierarchical fallback approach
5. **LLM-based Chunking**: Instruction-aware segmentation

### Embedding Models Tested
1. **OpenAI**: text-embedding-3-small, text-embedding-3-large
2. **E5**: intfloat/e5-base, intfloat/e5-large
3. **BGE**: BAAI/bge-small-en-v1.5, BAAI/bge-base-en-v1.5

## Compliance and Ethics

### Legal Compliance
- ✅ Respects robots.txt
- ✅ Follows website Terms & Conditions
- ✅ Only accesses publicly available content
- ✅ No user data or gated content accessed

### Data Usage
- Purpose: Customer support automation
- Scope: Public business and help documentation only
- Retention: Data stored locally for RAG system
- Sharing: Not redistributed, used only for this project

## Data Statistics

### Collection Statistics
- **Start Date**: [To be filled]
- **End Date**: [To be filled]
- **Total Collection Time**: [To be filled]
- **Success Rate**: [To be calculated]

### Content Statistics
- **Business Pages**: [To be counted]
- **FAQ Items**: [To be counted]
- **Help Articles**: [To be counted]
- **Total Documents**: [To be calculated]

### Processing Statistics
- **Chunks Generated**: [To be calculated per strategy]
- **Average Chunk Size**: [To be calculated]
- **Embedding Dimensions**: [Varies by model]

## Data Access

### File Structure
```
data/
├── scraped/           # Raw scraped HTML/text
├── processed/         # Cleaned and chunked data
├── chroma_db/        # Vector database
└── evaluation/       # Test datasets
```

### Data Formats
- **Raw Data**: HTML files, JSON metadata
- **Processed Data**: JSON with chunks and metadata
- **Vector Data**: ChromaDB collections
- **Evaluation Data**: JSON test sets

## Updates and Maintenance

### Data Freshness
- **Last Updated**: [To be filled]
- **Update Frequency**: As needed for accuracy
- **Version Control**: Git-tracked data processing scripts

### Quality Assurance
- Regular validation of scraped content
- Monitoring for website structure changes
- Periodic re-scraping for updated content

## Contact Information

For questions about this dataset or data collection process:
- **Project Repository**: [GitHub URL]
- **Maintainer**: [Your Name]
- **Last Updated**: [Date]
