import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_groq import ChatGroq
import time
import re
from urllib.parse import quote_plus

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def get_headers():
    """Returns request headers that mimic a real browser."""
    return {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    }

def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def fetch_article_content(url: str) -> str:
    """Fetch and extract main content from an article."""
    try:
        print(f"      📖 Reading: {url[:60]}...")
        time.sleep(1.5)
        
        response = requests.get(url, headers=get_headers(), timeout=15, allow_redirects=True)
        
        if response.status_code != 200:
            print(f"      ❌ Status: {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
            tag.decompose()
        
        # Try multiple strategies to find content
        content = ""
        
        # Strategy 1: Find article tag
        article = soup.find('article')
        if article:
            paragraphs = article.find_all('p', limit=10)
            content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
        
        # Strategy 2: Find main content div
        if not content:
            main_content = soup.find('main') or soup.find('div', {'role': 'main'})
            if main_content:
                paragraphs = main_content.find_all('p', limit=10)
                content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
        
        # Strategy 3: Look for content/body divs
        if not content:
            content_divs = soup.find_all('div', class_=re.compile(r'(content|body|article|post)', re.I), limit=3)
            for div in content_divs:
                paragraphs = div.find_all('p', limit=10)
                temp_content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
                if len(temp_content) > len(content):
                    content = temp_content
        
        if content:
            content = clean_text(content)
            print(f"      ✅ Extracted {len(content)} chars")
            return content[:2000]  # Limit to 2000 chars
        
        print(f"      ⚠️ No content found")
        return None
        
    except Exception as e:
        print(f"      ❌ Error: {str(e)[:50]}")
        return None

def search_google_custom(query: str) -> list:
    """
    Search Google for car reviews (more reliable than direct site search).
    """
    try:
        # Create a focused search query
        search_terms = f"{query} car review"
        encoded_query = quote_plus(search_terms)
        
        # Use Google search with site restriction
        url = f"https://www.google.com/search?q={encoded_query}+site:caranddriver.com+OR+site:carwow.co.uk"
        
        print(f"   🔍 Google Search: {search_terms}")
        time.sleep(1)
        
        response = requests.get(url, headers=get_headers(), timeout=15)
        
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        results = []
        
        # Find search result divs
        search_results = soup.find_all('div', class_='g', limit=5)
        
        for result in search_results:
            # Get title
            title_tag = result.find('h3')
            if not title_tag:
                continue
            
            title = title_tag.get_text(strip=True)
            
            # Get link
            link_tag = result.find('a', href=True)
            if not link_tag:
                continue
            
            link = link_tag['href']
            
            # Clean Google redirect URL
            if '/url?q=' in link:
                link = link.split('/url?q=')[1].split('&')[0]
            
            # Only include relevant domains
            if 'caranddriver.com' in link or 'carwow.co.uk' in link:
                source = 'Car and Driver' if 'caranddriver' in link else 'Carwow'
                results.append({
                    'title': title,
                    'link': link,
                    'source': source
                })
        
        print(f"   ✅ Found {len(results)} Google results")
        return results
        
    except Exception as e:
        print(f"   ⚠️ Google search error: {e}")
        return []

def search_caranddriver_direct(query: str) -> list:
    """Direct search on Car and Driver website."""
    try:
        # Clean query
        search_query = query.lower()
        search_query = re.sub(r'\b(review|reviews?|test|comparison)\b', '', search_query).strip()
        
        # Format for URL
        url_query = quote_plus(search_query)
        url = f"https://www.caranddriver.com/search?q={url_query}"
        
        print(f"   🔍 Car and Driver search: {url}")
        time.sleep(1)
        
        response = requests.get(url, headers=get_headers(), timeout=15)
        
        if response.status_code != 200:
            print(f"   ❌ Status: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        results = []
        
        # Find all links that look like articles
        links = soup.find_all('a', href=re.compile(r'/(reviews?|cars?|news)/'))
        
        seen_urls = set()
        for link in links[:10]:
            href = link.get('href', '')
            
            # Make absolute URL
            if href.startswith('/'):
                href = f"https://www.caranddriver.com{href}"
            
            # Skip duplicates
            if href in seen_urls:
                continue
            seen_urls.add(href)
            
            # Get title
            title = link.get_text(strip=True)
            
            # Skip if title is too short or generic
            if len(title) < 15 or title.lower() in ['read more', 'see more', 'learn more']:
                continue
            
            results.append({
                'title': title,
                'link': href,
                'source': 'Car and Driver'
            })
        
        print(f"   ✅ Found {len(results)} direct results")
        return results
        
    except Exception as e:
        print(f"   ⚠️ Direct search error: {e}")
        return []

@tool
def car_review_tool(query: str) -> str:
    """
    Fetches comprehensive car reviews and comparisons from multiple sources.
    
    Examples:
    - "BMW 5 Series 2025 review"
    - "Best luxury SUV"
    - "Compare BMW X5 vs Mercedes GLE"
    
    Args:
        query: Car review question or comparison request
        
    Returns:
        Detailed review summary with AI analysis
    """
    try:
        print(f"\n🚗 Car Review Search: '{query}'")
        print("="*60)
        
        # Try multiple search strategies
        all_results = []
        
        # Strategy 1: Google search (most reliable)
        google_results = search_google_custom(query)
        all_results.extend(google_results)
        
        # Strategy 2: Direct website search (if Google didn't work)
        if len(all_results) < 2:
            direct_results = search_caranddriver_direct(query)
            all_results.extend(direct_results)
        
        # Remove duplicates
        seen_links = set()
        unique_results = []
        for result in all_results:
            if result['link'] not in seen_links:
                seen_links.add(result['link'])
                unique_results.append(result)
        
        if not unique_results:
            print("   ❌ No results found")
            return f"""I couldn't find specific reviews for '{query}'. 

Here's what you can try:
1. Search directly: https://www.caranddriver.com/search?q={quote_plus(query)}
2. Try a different query (e.g., "2024 BMW 5 Series review")
3. Visit https://www.carwow.co.uk for UK reviews

Would you like me to help with something else about this car?"""
        
        print(f"\n📚 Found {len(unique_results)} articles. Fetching content...")
        
        # Fetch content from top 3 articles
        detailed_reviews = []
        for idx, result in enumerate(unique_results[:3], 1):
            print(f"\n   [{idx}/3] {result['title'][:50]}...")
            content = fetch_article_content(result['link'])
            
            if content and len(content) > 200:
                detailed_reviews.append({
                    'title': result['title'],
                    'link': result['link'],
                    'source': result['source'],
                    'content': content
                })
        
        if not detailed_reviews:
            # Fallback: Just provide links
            print("   ⚠️ Couldn't extract content, providing links")
            response_text = f"🚗 **Reviews for '{query}':**\n\n"
            response_text += "I found these relevant articles:\n\n"
            
            for i, result in enumerate(unique_results[:5], 1):
                response_text += f"{i}. **{result['title']}**\n"
                response_text += f"   Source: {result['source']}\n"
                response_text += f"   🔗 {result['link']}\n\n"
            
            response_text += "\n💡 Click the links above to read the full reviews."
            return response_text
        
        # Create AI summary from extracted content
        print(f"\n🤖 Generating AI summary from {len(detailed_reviews)} articles...")
        
        context = f"User asked about: {query}\n\n"
        for idx, review in enumerate(detailed_reviews, 1):
            context += f"=== Review {idx}: {review['title']} ({review['source']}) ===\n"
            context += f"{review['content']}\n\n"
        
        # Detect query type
        is_comparison = any(word in query.lower() for word in ['vs', 'versus', 'compare', 'or', 'better'])
        is_recommendation = any(word in query.lower() for word in ['best', 'recommend', 'should i', 'which', 'top'])
        
        if is_comparison:
            prompt = f"""Based on these car reviews, provide a comparison for: "{query}"

{context}

Create a comprehensive comparison including:
1. Brief overview of each car
2. Key differences
3. Pros and cons of each
4. Which is better for different use cases
5. Final recommendation

Keep it conversational and helpful (400 words max)."""
        elif is_recommendation:
            prompt = f"""Based on these reviews, provide recommendations for: "{query}"

{context}

Provide:
1. Overview of top options
2. Key features and strengths
3. Who each option is best for
4. Your recommendation

Keep it helpful and conversational (400 words max)."""
        else:
            prompt = f"""Based on these car reviews, create a comprehensive summary for: "{query}"

{context}

Include:
1. Overview and key highlights
2. Main strengths and weaknesses
3. Performance, features, and value
4. Who this car is best for
5. Final verdict

Keep it conversational and informative (400 words max)."""
        
        llm_response = llm.invoke(prompt)
        ai_summary = llm_response.content
        
        # Format final response
        response_text = f"🚗 **{query}**\n\n"
        response_text += f"{ai_summary}\n\n"
        response_text += "---\n\n"
        response_text += "📚 **Sources:**\n"
        
        for idx, review in enumerate(detailed_reviews, 1):
            response_text += f"{idx}. {review['title']}\n"
            response_text += f"   ({review['source']}) - {review['link']}\n"
        
        print("   ✅ Summary generated successfully")
        return response_text
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return f"I encountered an error searching for '{query}'. Please try: https://www.caranddriver.com/search?q={quote_plus(query)}"

