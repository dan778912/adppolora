import argparse  
import random
import torch
from torch import nn
from sentence_transformers import SentenceTransformer

class SimilarityLoss(nn.Module):
    def __init__(
        self,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        alpha=1.0,
        beta=1.0,
        gamma=1.0
    ):
        """
        Initialize the loss function and embedding model.

        Args:
            embedding_model_name: Name of the sentence transformer model.
            alpha: Weight for the l1 loss (query vs rewritten vs original).
            beta: Weight for the l2 loss (triplet sampling loss).
            gamma: Weight for the l3 loss (preservation loss).
        """
        super(SimilarityLoss, self).__init__()
        self.encoder = SentenceTransformer(embedding_model_name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_l1(self, query, original_doc, rewritten_doc):
        # Encode query and documents
        query_emb = self.encoder.encode([query], convert_to_tensor=True).squeeze(0)
        orig_emb = self.encoder.encode([original_doc], convert_to_tensor=True).squeeze(0)
        gen_emb = self.encoder.encode([rewritten_doc], convert_to_tensor=True).squeeze(0)
        
        # Calculate similarities
        sim_query_gen = torch.nn.functional.cosine_similarity(query_emb, gen_emb, dim=0)
        sim_query_orig = torch.nn.functional.cosine_similarity(query_emb, orig_emb, dim=0)
        
        return -(sim_query_gen - sim_query_orig)  # maximize improvement in similarity


    def compute_l2(self, query, top_k_docs, rewritten_doc):
        # Encode query and rewritten document
        q_emb = self.encoder.encode([query], convert_to_tensor=True).squeeze(0)
        gen_emb = self.encoder.encode([rewritten_doc], convert_to_tensor=True).squeeze(0)
        
        # Randomly sample up to 3 documents from top_k_docs
        sampled_docs = random.sample(top_k_docs, min(3, len(top_k_docs)))
        sampled_docs_embs = [self.encoder.encode([doc], convert_to_tensor=True).squeeze(0) for doc in sampled_docs]
        
        # Calculate average similarity between query and random docs
        avg_sim_randoms = torch.stack([
            torch.nn.functional.cosine_similarity(q_emb, emb, dim=0) 
            for emb in sampled_docs_embs
        ]).mean() if sampled_docs_embs else torch.tensor(0.0, device=gen_emb.device)
        
        # Calculate similarity between query and rewritten doc
        sim_gen = torch.nn.functional.cosine_similarity(q_emb, gen_emb, dim=0)
        
        return -(sim_gen - avg_sim_randoms)  # Maximize how much closer rewritten_doc is to the query

    def compute_l3(self, original_doc, rewritten_doc):
        # Encode original and rewritten documents
        orig_emb = self.encoder.encode([original_doc], convert_to_tensor=True).squeeze(0)
        gen_emb = self.encoder.encode([rewritten_doc], convert_to_tensor=True).squeeze(0)
        
        # Calculate preservation loss (1 - similarity)
        return 1 - torch.nn.functional.cosine_similarity(orig_emb, gen_emb, dim=0)


    def forward(self, query, original_doc, rewritten_doc, top_k_docs):
        """
        Compute the combined loss: L = αl1 + βl2 + γl3

        Args:
            query: A string (the user query).
            original_doc: Original ad text.
            rewritten_doc: Rewritten ad text.
            top_k_docs: List of top-k ad texts (strings).

        Returns:
            Scalar loss.
        """
        l1 = self.compute_l1(query, original_doc, rewritten_doc)
        l2 = self.compute_l2(query, top_k_docs, rewritten_doc)
        l3 = self.compute_l3(original_doc, rewritten_doc)

        return self.alpha * l1 + self.beta * l2 + self.gamma * l3

# testing
query = "what is a good piece of sound equipment?"

# Original and rewritten ad
original_doc = {
    "user_query": "behringer juno chorus",
    "title": "Behringer U-Phoria UMC22 USB Audio Interface 48kHz, 2-channel USB Audio Interface with 1 MIDAS Preamp, Phantom Power, and Instrument Input",
    "text": "Audio Interfaces - Recording quality audio in your Mac or Windows PC home studio is easy and rewarding, when you're recording with a Behringer U-Phoria UMC22. Any Sales Engineer here at Sweetwater, will be happy to tell you that you don't need the fanciest equipment out there to capture your ideas and bring your music to life, and the U-Phoria UMC22 USB audio interface gives you everything you need to get started. Onboard, you'll find a genuine MIDAS microphone preamp, which delivers truly impressive sound. There's also a dedicated instrument input onboard the UMC22, so you can record yourself singing as you play guitar or keys. And since the Behringer U-Phoria UMC22 includes a full copy of Tracktion, you're ready to start recording, right from day one.Real MIDAS microphone preamplifier technology onboardOne of the things that really makes the Behringer U-Phoria UMC22 stand out is its genuine MIDAS preamplifier technology. Known throughout the world of live sound for delivering ultra clear sound with plenty of headroom, MIDAS preamps are among the most popular mic pres on Earth. What's more, these preamps come standard with +48V phantom power onboard, so you can use your choice of quality condenser microphones with your UMC22. Call your Sweetwater Sales Engineer, and turn this U-Phoria UMC22 USB audio interface into a killer package deal with the perfect microphones and monitors for your complete recording rig.Behringer U-Phoria UMC22 USB Audio Interface Features: Quality 2-channel USB recording interface for your Mac or Windows PC. Combo input with MIDAS preamp lets you plug in any microphone or line-level gear. +48V phantom power lets you use studio condenser microphones. Dedicated instrument-level input accommodates your guitar or bass. Headphone and stereo 1/4\" outputs provide easy monitoring. Includes Tracktion DAW and 150 downloadable instrument/effect plug-ins. Start recording with a Behringer U-Phoria UMC22 USB audio interface! USB Audio Interfaces",
    "url": "sweetwater.com",
    "seller": "Sweetwater",
    "brand": "Behringer",
    "source": "microsoft_commercial_ads",
    "ad_id": "a767622a-e5f8-48f9-b026-e960bb488e96"
}
rewritten_doc = {
    "user_query": "behringer juno chorus",
    "title": "Behringer U-Phoria UMC22 USB Audio Interface 48kHz, 2-channel USB Audio Interface with 1 MIDAS Preamp, Phantom Power, and Instrument Input",
    "text": "Audio Interfaces - Recording quality audio in your Mac or Windows PC home studio is easy and rewarding, when you're recording with a Behringer U-Phoria UMC22. Any Sales Engineer here at Sweetwater, will be happy to tell you that you don't need the fanciest equipment out there to capture your ideas and bring your music to life, and the U-Phoria UMC22 USB audio interface gives you everything you need to get started. Onboard, you'll find a genuine MIDAS microphone preamp, which delivers truly impressive sound. There's also a dedicated instrument input onboard the UMC22, so you can record yourself singing as you play guitar or keys. And since the Behringer U-Phoria UMC22 includes a full copy of Tracktion, you're ready to start recording, right from day one.Real MIDAS microphone preamplifier technology onboardOne of the things that really makes the Behringer U-Phoria UMC22 stand out is its genuine MIDAS preamplifier technology. Known throughout the world of live sound for delivering ultra clear sound with plenty of headroom, MIDAS preamps are among the most popular mic pres on Earth. What's more, these preamps come standard with +48V phantom power onboard, so you can use your choice of quality condenser microphones with your UMC22. Call your Sweetwater Sales Engineer, and turn this U-Phoria UMC22 USB audio interface into a killer package deal with the perfect microphones and monitors for your complete recording rig.Behringer U-Phoria UMC22 USB Audio Interface Features: Quality 2-channel USB recording interface for your Mac or Windows PC. Combo input with MIDAS preamp lets you plug in any microphone or line-level gear. +48V phantom power lets you use studio condenser microphones. Dedicated instrument-level input accommodates your guitar or bass. Headphone and stereo 1/4\" outputs provide easy monitoring. Includes Tracktion DAW and 150 downloadable instrument/effect plug-ins. Start recording with a Behringer U-Phoria UMC22 USB audio interface! USB Audio Interfaces",
    "url": "sweetwater.com",
    "seller": "Sweetwater",
    "brand": "Behringer",
    "source": "microsoft_commercial_ads",
    "ad_id": "a767622a-e5f8-48f9-b026-e960bb488e96"
}

# Initialize the loss function (you can adjust alpha, beta, gamma if desired)
loss_fn = SimilarityLoss(alpha=1.0, beta=1.0, gamma=1.0)

# Simulate top-k retrieved documents (e.g. from a retrieval engine)
top_k_docs = [
    {
    "user_query": "countryman microphones",
    "title": "Countryman E6XDP5B1 E6 Flex Directional Headworn Microphone for Phantom-Powered, Black",
    "text": "The E6 Flex Directional Earset combines the best of the classic E6 directional earset with the best of the directional E6i. The slim, springy ear section grips securely like an E6 classic, while the flexible front boom takes abuse like the E6i and provides precise placement and easy reshaping. The ultra miniature directional element delivers exceptional sound quality and rejects surrounding noise or feedback from stage monitors and nearby speakers. As part of the Countryman E6 Directional line, the E6 Flex Directional features swapable reinforced cables, versatile skin tone options, changeable caps for cardioid or hypercardioid patterns, and rugged construction. It weighs less than one-tenth ounce and virtually disappears against the skin, so performers forget they're even wearing a mic.The fit of the classic E6 meets the flexibility of the E6i. The front boom is like an E6i, the ear section is like the classic E6, the result is the best fitting, easiest to use earset available. Country Earsets are the smallest, lightest, and least visible head-worn microphones. The E6 Flex comes in four skin tones to disappear against the face. Changeable protective caps let you switch between cardioid and hypercardioid patterns in seconds.Frequency response is better than 30Hz to 15kHz with smooth off-axis response and >100dB dynamic range. Countryman Earsets sound like world-class, full-size performance mics but the performer has complete freedom. The E6i is exceptionally resistant to makeup, sweat and moisture when used with the supplied protective caps. The skin-colored, almost unbreakable boom resists water, sweat, and makeup and can be bent and re-bent many times to fit different performers. The front boom of the E6 Flex is slightly firmer than the E6i, so it's easily shaped right on the performer's face and smooths out for a professional look.The E6 Flex comes in three sensitivities for different speaking or singing styles, with up to 140 dB SPL capability. With other microphones a worn cable requires purchasing a...",
    "url": "fullcompass.com",
    "seller": "Full Compass Systems",
    "brand": "Countryman",
    "source": "microsoft_commercial_ads",
    "ad_id": "e2ba2905-e627-4c3a-a235-a3b3d84c4dbc"},
    
    {
    "user_query": "behringer juno chorus",
    "title": "Behringer U-Phoria UMC22 USB Audio Interface 48kHz, 2-channel USB Audio Interface with 1 MIDAS Preamp, Phantom Power, and Instrument Input",
    "text": "Audio Interfaces - Recording quality audio in your Mac or Windows PC home studio is easy and rewarding, when you're recording with a Behringer U-Phoria UMC22. Any Sales Engineer here at Sweetwater, will be happy to tell you that you don't need the fanciest equipment out there to capture your ideas and bring your music to life, and the U-Phoria UMC22 USB audio interface gives you everything you need to get started. Onboard, you'll find a genuine MIDAS microphone preamp, which delivers truly impressive sound. There's also a dedicated instrument input onboard the UMC22, so you can record yourself singing as you play guitar or keys. And since the Behringer U-Phoria UMC22 includes a full copy of Tracktion, you're ready to start recording, right from day one.Real MIDAS microphone preamplifier technology onboardOne of the things that really makes the Behringer U-Phoria UMC22 stand out is its genuine MIDAS preamplifier technology. Known throughout the world of live sound for delivering ultra clear sound with plenty of headroom, MIDAS preamps are among the most popular mic pres on Earth. What's more, these preamps come standard with +48V phantom power onboard, so you can use your choice of quality condenser microphones with your UMC22. Call your Sweetwater Sales Engineer, and turn this U-Phoria UMC22 USB audio interface into a killer package deal with the perfect microphones and monitors for your complete recording rig.Behringer U-Phoria UMC22 USB Audio Interface Features: Quality 2-channel USB recording interface for your Mac or Windows PC. Combo input with MIDAS preamp lets you plug in any microphone or line-level gear. +48V phantom power lets you use studio condenser microphones. Dedicated instrument-level input accommodates your guitar or bass. Headphone and stereo 1/4\" outputs provide easy monitoring. Includes Tracktion DAW and 150 downloadable instrument/effect plug-ins. Start recording with a Behringer U-Phoria UMC22 USB audio interface! USB Audio Interfaces",
    "url": "sweetwater.com",
    "seller": "Sweetwater",
    "brand": "Behringer",
    "source": "microsoft_commercial_ads",
    "ad_id": "a767622a-e5f8-48f9-b026-e960bb488e96"}, 
    
    {
    "user_query": "best acoustic electric guitars",
    "title": "Martin DJR Acoustic Electric Guitar Natural with Gig Bag",
    "text": "Martin introduces a new body size with the solid wood Dreadnought Junior, which is fashioned for player comfort, clear powerful tone and easy action. The affordably-priced Dreadnought Junior is ideal for smaller players, students, travelers, or anyone who aspires to the clarity and depth of tone that has defined Martin instruments for more than 180 years. Fashioned for player comfort, clear powerful tone and easy action, the Dreadnought Junior is reduced to approximately 15/16 of the full Martin 14-fret Dreadnought dimension (14 1/4-inch width at the lower bout) with an expressive 24 inch scale length. A solid Sitka spruce top features scalloped 1/4-inch high performance X-bracing with a single asymmetrical tone bar. The back and sides are bookmatched from sapele, and Richlite, an ebony alternative with similar hardness and appearance, is chosen for the fingerboard and bridge. The Dreadnought Junior (aka D Jr) comes factory-equipped with Fishmans Sonitone sound reinforcement system and includes a nylon gig bag. Martin Dreadnought Junior Acoustic Electric Guitar Features Dreadnought Junior Body Size (15/16 of the full Martin 14-fret Dreadnought) Solid Sitka Spruce Top Solid Sapele Back and Sides Select Hardwood Neck with Richlite Fingerboard Fishman Sonitone Electronics Gig Bag Included CALIFORNIA PROPOSITION 65 WARNING WARNING: Cancer and Reproductive Harm - www.P65Warnings.ca.gov.",
    "url": "americanmusical.com",
    "seller": "AmericanMusical.com",
    "brand": "Martin",
    "source": "microsoft_commercial_ads",
    "ad_id": "29c2202b-3373-49c1-aeac-28acb0e6f562"}
]

# Call the forward function to compute the loss
loss = loss_fn(query, original_doc, rewritten_doc, top_k_docs)

# Print the scalar loss value
print("Loss:", loss.item())