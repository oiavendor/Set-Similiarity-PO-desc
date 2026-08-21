CONTEXT = {
    1: "The procurement team has tracked projects over the years by description. Now they want to create a project category for easier searching and indexing.",
    2: "If an appropriate category exist in the list. Respond with the category from the list",
    3: "If an appropriate category does not exist in the list. Generate a new broad category that the project falls under",
    4: "Be broader with the category.",
    5: "These projects are within National University of Singapore, hence these categories should make sense in a University context.",
}

CONSTRAINTS = {
    1: "Do not provide any additional explanation beyond the category.",
    2: "Do not add additional keys to the answer, for example: 'category:' or 'category name:'.",
    3: "Do not use ',' in category value.",
    4: "Category must be a single category.",
    5: "Do not have hybrid category consisting of 2 or more sub categories.",
}

EXAMPLE = {
    1: "Project Description: 'SURGICAL TOOLING FOR NUS MITRAL AND TRICUSPID VALVE PROJECTS',Could be categorized as Laboratory Equipment Sourcing.",
    2: "Project Description: 'ENGAGEMENT OF NG DA XUAN FOR PROVISION OF SERVICE IN RESEARCH CONSULTATION AND ON-GOING SUPERVISION IN FIT FACILITATION', Could be categorized as Research Consultation",
    3: "Project Description: 'WARRANTY AND ASSURANCE PLAN COVERAGE FOR CHROMIUM X MACHINE', Could be categorized as Laboratory Equipment Sourcing."
}