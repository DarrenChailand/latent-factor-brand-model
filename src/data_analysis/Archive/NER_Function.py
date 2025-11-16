# pip install spacy
# python -m spacy download en_core_web_sm

import spacy

nlp = spacy.load("en_core_web_sm")

def extract_features_and_adjectives(text):
    doc = nlp(text)

    features = []
    p_features = []
    adjectives = []

    for token in doc:
        # Feature candidates: nouns
        if token.pos_ in ["NOUN"]:
            features.append(token.lemma_.lower())

        if token.pos_ in ["PROPN"]:
            p_features.append(token.lemma_.lower())

        # Adjective candidates
        if token.pos_ == "ADJ":
            adjectives.append(token.lemma_.lower())

    return {
        "features": features,
        "p_features": p_features,
        "adjectives": adjectives

    }

# Example text
text = """
Apple, Samsung, and Google are three prominent technology companies that produce a wide range of products, including smartphones, tablets, smartwatches, and more. While they share some similarities, each company has its unique strengths and weaknesses. Here's a comparison of their key differences in terms of product quality and reliability:\n\n**Product Quality:**\n\n1. **Design and Build**: Apple is known for its minimalist and premium designs, often with attention to detail and high-quality materials. Samsung, on the other hand, offers a wide range of devices with varying levels of design refinement.\n2. **Display**: Google's Pixel series (e.g., Pixel 6) and Samsung's Galaxy S series are highly regarded for their excellent display quality, with vibrant colors, sharp text, and impressive brightness.\n3. **Performance**: Apple's A-series processors are generally considered more reliable and efficient, while Samsung's Exynos processors may have a slight edge in terms of performance.\n\n**Reliability:**\n\n1. **Water Resistance**: Google devices (e.g., Pixel series) often come with IP68 water resistance ratings, making them more durable than iPhones.\n2. **Battery Life**: Samsung's Galaxy S series typically offers longer battery life compared to Apple devices.\n3. **Software Updates**: Both companies release regular software updates, but Samsung's updates are generally faster and more frequent.\n\n**Customer Support:**\n\n1. **Apple Support**: Apple's customer support is often praised for its extensive online resources, expert guides, and exceptional phone support via Apple Store representatives.\n2. **Samsung Support**: Samsung's customer support has improved in recent years, with dedicated apps for various devices and better phone support through the Samsung Support website.\n\n**Pricing:**\n\n1. **Apple**: iPhones tend to be pricier than Samsung devices, especially when it comes to flagship models.\n2. **Samsung**: Offers a wider range of affordable options, including mid-range devices that rival Apple prices.\n\n**Battery Life:**\n\n1. **Google Pixel series**: Excellent battery life, with some devices offering up to two days of usage on a single charge.\n2. **Samsung Galaxy S series**: Long-lasting batteries, but may not match the Pixel's efficiency.\n3. **Apple iPhone 14 Pro**: Similar battery life to Samsung's flagship models.\n\n**Security:**\n\n1. **Google**: Google's software is generally considered more secure than Apple's, thanks to its open-source nature and stricter app review process.\n2. **Samsung**: Also prioritizes security, but some users may notice minor security vulnerabilities in their devices.\n\nIn summary:\n\n* Apple excels in product design, display quality, and performance, while also offering excellent customer support.\n* Samsung focuses on delivering high-quality displays, water resistance, and battery life, often at a lower price point than iPhones.\n* Google prioritizes software updates, user experience, and security, with devices that are highly regarded for their overall reliability.\n\nUltimately, the choice between Apple, Samsung, or Google depends on your individual preferences and priorities. If you value design, performance, and ecosystem integration, Apple might be the best fit. For those seeking a more affordable option with great display quality, water resistance, and battery life, Samsung could be an excellent choice. Google devices are often preferred by those who prioritize software updates, user experience, and security.
"""

result = extract_features_and_adjectives(text)
print(result)
