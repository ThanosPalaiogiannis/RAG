import bs4
from rag.ingestion.fetch import fetch_html 
from dataclasses import dataclass
import re 

@dataclass
class ParagraphBlock:
    text: str
    
@dataclass
class HeadingBlock:
    text: str
    level: int
    
@dataclass
class CodeBlock:
    code: str
    language: str
    
Block = HeadingBlock | ParagraphBlock | CodeBlock

@dataclass 
class ParsedDocument:
    url: str
    blocks: list[Block]
    
def extract_blocks_from_html(main) -> list[Block]:  
    tags = main.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "pre"])
    blocks = []
    for tag in tags:
        if tag.name.startswith("h"):
            level = int(tag.name[1])
            blocks.append(HeadingBlock(text=tag.get_text(strip=True), level=level))
        elif tag.name == "p":
            text = re.sub(r"\s+([,.;:?!])", r"\1", tag.get_text(" ",strip=True))
            if text:
                blocks.append(ParagraphBlock(text=text))
        elif tag.name == "pre":
            code_tag = tag.find("code")
            if code_tag:
                language = code_tag.get("data-language","")
                line_numbers = code_tag.find(class_="syntax-highlighter-line-numbers")
                if line_numbers:
                    line_numbers.decompose()
                code_lines = code_tag.find_all(class_="syntax-highlighter-line")
                if code_lines:
                    code = "".join(line.get_text(strip=False) for line in code_lines)
                    blocks.append(CodeBlock(code=code, language=language))
    return blocks
    
def parse_html(html: str, url: str) -> ParsedDocument:
    soup = bs4.BeautifulSoup(html, "lxml")
    main = soup.find("main")
    if main is None:
        raise ValueError(f"No <main> tag found in page at {url}")
    for aside in main.find_all("aside"):
        aside.decompose()
    blocks = extract_blocks_from_html(main)
    return ParsedDocument(url=url, blocks=blocks)