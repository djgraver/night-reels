"""The example dreams used for the first round of renders."""
DREAMS = {
    "hallway": "I was back in my grandmother's house but the hallway just kept going, doors on both sides, "
               "and warm light was coming from under the door at the very end. Someone was standing in the hall but I couldn't see who.",
    "beach": "I was walking along a beach at night. The sea was completely still, like glass, and there was a huge moon "
             "sitting just above the water. It felt calm but strange.",
    "mountain": "It was crystal clear, I remember every detail. I was standing on a mountain ridge in bright morning sun, "
                "looking down over a valley full of lakes.",
    "station": "In black and white, like an old film. An empty train station, and a single man standing on the platform "
               "under a big clock. It was silent.",
    "flooded_city": "I was swimming down a flooded street at dusk. The streetlights were underwater and still glowing, "
                    "and everything was purple.",
    "forest": "A forest, really hazy, I couldn't see very far. In the middle of the trees there was a red door, "
              "just standing there on its own, slightly open with light behind it.",
    "stairs": "I was floating up a white staircase into the clouds. The colours were so vivid and saturated, "
              "like a painting.",
}

# Round 2: fog tone. Each reuses the frames of an existing scene; only the dream pass differs.
TONE_DREAMS = {
    "hallway_dark": ("hallway", "It was a nightmare. I was in my grandmother's hallway again but it was filling with black smoke, "
                                "the lights were going out one by one and something was behind the door at the end."),
    "forest_shift": ("forest", "I was walking through a misty forest, it was peaceful and soft and warm. Then suddenly it went dark, "
                               "the fog turned black and I was terrified of the red door."),
    "beach_shift": ("beach", "At first everything was black, I couldn't see anything at all. Then the fog lifted and the moon came out "
                             "over a calm sea, still like glass."),
}
