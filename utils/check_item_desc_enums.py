CONTEXT = {
    1: "The procurement team is performing a check on whether item description of a purchase order is related to any of the 3 item categories.",
    # 3: "If the item description is for a gown and the item category is for domestic assistance and services, it could mean that the gown was cleaned as part of the laundry sub-service of domestic assistance.",
    # 4: "If the item description is 'Insertion sleeve E' and the item category is related to laboratory it should be a match as insertion sleeves are used in the laboratory.",
    # 5: "If the item description is for food and the categories is roughly related to events, it should be a match as it means the events procured a catering service.",
    # 6: "If item description is for tickets and the categories are related to any services that could use tickets, it should match.",
    # 7: "If item description is for dimensions or technical specifications of products, it should be a match if it makes sense. 'A4' matches for 'Graphic Design' or 'Art' categories as it is the dimensions for paper.",
    # 8: "If item description is for a product or service linked to the category it should match if it makes sense.",
    # 9: "'Consultancy Fees' should match 'Construction' or 'Maintainence and Support' categories as the consulation is for these categories.",
    # 10: "'Blanket' should match 'Laundry type combined washing or drying machines' category as blanket is washed as laundry."
}

CONSTRAINTS = {
    1: "Do not provide any additional explanation beyond 'Yes,{Explanation of why item description matches any of the 3 categories}'(not more than 50 words) or 'No,{Explanation of why item description does not match all of the 3 categories}'(not more than 50 words)",
    2: "Always try to relate the item description to the any categories and only respond with No if it is impossible to relate it to all categories.",
    3: "Do not add additional keys to the answer, for example: 'mismatch explanation:' or 'explanation for mismatch:'.",
    # 4: "Give a general explanation without highlighting any specific products or services",
    5: "The explanation must start with: 'Item matches the category because' or 'Item does not match with the category because'",
    6: "Do not assume what a product or service could be.",
    7: "If there is not enough information about product or service, it must not match any categories.",
    8: "Be more lenient with food and drinks as certain foods like oats and cereals can be considered drinks.",
    # 9: "Explanations must have context in them related to description and categories.",
}

EXAMPLE = {
    1: "'5-FLUOROINDOLE,5 GRAM,SOLID,399-52-0' as the item description can be accurately classified in the item category 'Aliphatic and aromatic compounds', and therefore is accurate.",
    2: "'Seagate 2TB external 2.5 HDD' as the item description can be accurately classified in the item category 'Computer Equipment and Accessories', and therefore is accurate.",
    3: "'Quotation no:Q-355543' as the item description can not be accurately classified in the item category 'Material packing and handling', and therefore is not accurate.",
    4: "'18:2 (CIS) PC (DLPC)' or '16:0-12:0 NBD PC' or '160908 35S:RUBY' as the item description cannot be accurately classified in the item category 'Chemicals' or 'Laboratory Items' as it is not 100% certainty - a chemical compound."
}