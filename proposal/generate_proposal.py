#!/usr/bin/env python3
"""
Generate LaTeX proposal document for DataMender project.
Based on the project requirements and documented plans.
"""

import os


def create_latex_proposal():
    """Create the DataMender project proposal as raw LaTeX."""
    
    latex_content = r"""
\documentclass[11pt]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{enumitem}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{hyperref}

% Custom colors
\definecolor{primaryblue}{RGB}{41, 128, 185}
\definecolor{accentgreen}{RGB}{39, 174, 96}
\definecolor{warningorange}{RGB}{230, 126, 34}
\definecolor{darkgray}{RGB}{52, 73, 94}

% Custom formatting
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlength{\baselineskip}{12pt}

% Header formatting
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\color{primaryblue}\textbf{DataMender Project Proposal}}
\fancyhead[R]{\color{darkgray}Group 8}
\fancyfoot[C]{\thepage}

\begin{document}

% Title section
\begin{center}
    {\Huge\color{primaryblue}\textbf{DataMender: Smart Cleaning for Large CSV/Parquet Files}}
    
    \vspace{0.3cm}
    {\Large\color{darkgray}Project Proposal}
    
    \vspace{0.5cm}
    \begin{tikzpicture}
        \draw[accentgreen, thick, rounded corners, fill=accentgreen!20] (-4,-0.6) rectangle (4,0.6);
        \node[darkgray, font=\large\bfseries] at (0,0) {AMS 560 / CSE 542 - Fall 2025};
    \end{tikzpicture}
    
    \vspace{0.5cm}
    {\color{darkgray}\textbf{Team Members:}} \\
    Ahmad Javadi Nezhad • Daniel Bazmandeh • Iliya Mirzaei • Nicholas Tardugno • Tamali Halder
    
    \vspace{0.3cm}
    {\color{darkgray}September 2025}
\end{center}

\vspace{0.5cm}

% Section 1: Problem Statement and Motivation
\section{\color{primaryblue}Problem Statement and Motivation}

Data cleaning stands as the most significant bottleneck in modern data science workflows. Research consistently shows that data scientists allocate 60-80\% of their working hours to data preparation tasks—time that could otherwise be spent on analysis, modeling, and extracting business insights. This problem intensifies dramatically when working with enterprise-scale datasets spanning multiple gigabytes, where traditional manual cleaning approaches become not just inefficient, but practically impossible.

The scale challenge compounds several fundamental issues. Modern datasets frequently contain inconsistent formatting across columns, missing values that follow complex patterns, and duplicate records that require sophisticated detection algorithms. Each of these problems demands careful analysis and domain expertise to resolve correctly, yet current tools force data scientists to manually craft cleaning rules for every unique scenario they encounter.

The consequences of poor data quality cascade throughout entire analytical pipelines. Inconsistent data leads to unreliable statistical analyses, biased machine learning models, and ultimately, flawed business decisions that can cost organizations millions of dollars. In an era where data-driven decision making has become a competitive necessity, organizations cannot afford the luxury of manual, error-prone data cleaning processes.

What the industry desperately needs is an intelligent system that can automatically identify data quality issues, suggest contextually appropriate cleaning strategies, and seamlessly integrate human expertise to validate and refine these suggestions. This represents a fundamental shift from reactive manual cleaning to proactive, AI-assisted data quality management.

% Section 2: Related Work and Current Limitations
\section{\color{primaryblue}Related Work and Current Limitations}

The emergence of large language models has catalyzed a new generation of data cleaning tools that promise to automate traditionally manual processes. Recent research has produced several notable advances: Cocoon [3] demonstrates how LLMs can perform semantic table profiling to understand data structure and meaning, AutoDCWorkflow [4] shows promise in automatically generating complete cleaning workflows, and LLMClean [1] introduces techniques for creating context-aware cleaning rules that adapt to specific data domains.

These developments represent significant progress toward automated data cleaning, yet they collectively expose three critical gaps that prevent widespread adoption in enterprise environments.

The first limitation concerns scale. Current LLM-based tools operate effectively on small to medium datasets but fail when confronted with the multi-gigabyte files common in production environments. Memory constraints, processing timeouts, and inadequate sampling strategies render these tools impractical for real-world big data scenarios.

The second gap involves validation and control. Existing tools provide limited mechanisms for human oversight, forcing users to choose between full automation (with attendant risks) and manual processes (with scalability constraints). This binary choice is particularly problematic in high-stakes environments where data quality errors can have significant business or regulatory consequences.

The third limitation addresses the fundamental challenge of LLM hallucination in data cleaning contexts. When language models generate cleaning rules, they may introduce subtle errors, inappropriate transformations, or rules that work well on sample data but fail catastrophically when applied to complete datasets. Current tools provide insufficient safeguards against these failure modes, making them unsuitable for production use.

These limitations create a clear market opportunity for a tool that combines the intelligence of LLM-powered automation with robust human oversight mechanisms, specifically designed to handle enterprise-scale data cleaning challenges.

% Section 3: Proposed Solution Architecture
\section{\color{primaryblue}DataMender: A Human-AI Collaborative Approach}

DataMender addresses these fundamental limitations through a carefully designed three-component architecture that balances automation efficiency with human oversight and control. Our approach recognizes that optimal data cleaning requires both machine intelligence for pattern recognition and human expertise for contextual validation.

\textbf{Intelligent Data Profiler:} The foundation of our system employs advanced profiling techniques built on optimized Pandas, NumPy, and Dask operations to efficiently analyze multi-gigabyte datasets without memory overflow. Our profiler goes beyond simple statistical summaries to identify complex data quality patterns, cross-column relationships, and domain-specific anomalies that inform intelligent rule generation. By using strategic sampling and incremental processing techniques, we can profile datasets that exceed available system memory while maintaining statistical validity.

\textbf{Multi-Model Rule Discovery Engine:} Rather than relying on a single LLM, our rule discovery engine leverages multiple language models including GPT-4, Claude-3, and specialized local models to generate diverse cleaning suggestions. Each model contributes unique strengths: GPT-4 excels at complex reasoning tasks, Claude-3 provides robust safety considerations, and local models offer privacy and cost advantages. Our ensemble approach generates confidence scores for each suggested rule based on data coverage, pattern consistency, and cross-model agreement, providing users with quantitative measures of rule reliability.

\textbf{Interactive Validation Interface:} The final component provides an intuitive yet powerful interface for reviewing, editing, and approving suggested cleaning rules before application. Our validation workflow includes side-by-side data previews, impact analysis showing exactly which records will be affected, and reversible operation logging that enables immediate rollback if issues arise. This interface transforms rule validation from a tedious bottleneck into an efficient quality assurance process.

This architecture creates a seamless workflow where machine intelligence handles the computationally intensive tasks of pattern identification and rule generation, while human expertise focuses on the high-value activities of validation, refinement, and strategic decision-making.

% Section 4: Technical Innovation and Risk Mitigation
\section{\color{primaryblue}Technical Innovation and Reliability Safeguards}

DataMender introduces several key innovations that advance the state-of-the-art in automated data cleaning while addressing the reliability concerns that have limited adoption of existing LLM-based tools.

\textbf{Scalable LLM Integration:} Our most significant technical contribution involves developing efficient strategies for applying LLM analysis to datasets that far exceed model context limits. We employ intelligent stratified sampling that preserves statistical properties across data subsets, adaptive chunking that maintains semantic coherence, and incremental rule refinement that allows models to build upon previous analyses. These techniques enable comprehensive LLM analysis of multi-gigabyte datasets using standard computational resources.

\textbf{Comprehensive Hallucination Mitigation:} We implement a multi-layered approach to ensure rule reliability and prevent the catastrophic failures that can result from LLM hallucination. Our strategy includes five complementary safeguards: quantitative confidence scoring based on data coverage and pattern consistency metrics, cross-validation between multiple LLM models to identify discrepancies, mandatory human review for all generated rules with clear approval workflows, comprehensive operation logging that enables immediate rollback of problematic transformations, and rigorous test data validation where rules are applied to representative samples before full dataset processing.

\textbf{Domain-Aware Rule Templates:} We provide pre-built rule templates optimized for common data domains including financial records, healthcare data, geographic information, and e-commerce transactions. These templates incorporate domain-specific validation rules, standard formatting conventions, and regulatory compliance requirements, significantly reducing the likelihood of generating inappropriate cleaning rules while accelerating the overall cleaning process.

\textbf{Performance Optimization:} Our system addresses the computational challenges of large-scale data cleaning through several optimization strategies. Memory-mapped file access minimizes RAM requirements, parallel processing pipelines maximize CPU utilization, and incremental transformation capabilities allow users to process datasets in manageable chunks while maintaining consistency across the entire cleaning workflow.

The combination of these innovations creates a robust, production-ready system that maintains the intelligence benefits of LLM-powered automation while providing the reliability guarantees required for enterprise deployment.

% Section 5: Competitive Analysis and Market Position
\section{\color{primaryblue}Competitive Landscape and Differentiation}

DataMender enters a market segment currently divided between free but limited tools and expensive enterprise solutions, with emerging LLM-based tools occupying an uncertain middle ground. Our competitive analysis reveals clear differentiation opportunities across multiple dimensions.

\begin{table}[h]
\centering
\footnotesize
\begin{tabular}{|l|p{2.5cm}|p{2.5cm}|p{3cm}|}
\hline
\textbf{Capability} & \textbf{OpenRefine} & \textbf{Trifacta/Alteryx} & \textbf{DataMender} \\
\hline
\textbf{Rule Generation} & Manual rule creation & Template-based suggestions & \textbf{Full LLM automation with confidence scoring} \\
\hline
\textbf{Scale Capacity} & Memory-constrained & Enterprise infrastructure & \textbf{Multi-gigabyte optimization with standard hardware} \\
\hline
\textbf{Validation Process} & Basic preview & Enterprise workflow & \textbf{Interactive AI-assisted validation} \\
\hline
\textbf{Error Prevention} & User experience & Basic rule testing & \textbf{Multi-model hallucination mitigation} \\
\hline
\textbf{Deployment Model} & Open source & Enterprise licensing & \textbf{Open source with enterprise features} \\
\hline
\textbf{Domain Intelligence} & Generic transformations & Industry templates & \textbf{LLM-powered domain awareness} \\
\hline
\end{tabular}
\caption{Competitive feature analysis across major data cleaning platforms}
\label{tab:comparison}
\end{table}

OpenRefine, while popular in academic and small-scale commercial contexts, fundamentally relies on manual rule creation and lacks the computational architecture needed for enterprise-scale datasets. Users must manually identify data quality issues and craft appropriate transformations, making it unsuitable for the automated workflows that modern organizations require.

Trifacta and Alteryx represent the current enterprise standard, offering sophisticated workflow management and integration capabilities. However, their rule generation remains largely manual or template-based, requiring significant user expertise and time investment. More critically, their enterprise pricing models place them out of reach for many organizations, particularly those in academic, nonprofit, or startup contexts.

Emerging LLM-based tools show promise but suffer from the scale and reliability limitations discussed earlier. Most operate as research prototypes rather than production-ready systems, lacking the robust error handling and validation workflows necessary for enterprise deployment.

DataMender uniquely combines advanced LLM automation with enterprise-grade reliability safeguards, all within an open-source framework that ensures broad accessibility. This positions us to capture demand from organizations seeking intelligent automation without the prohibitive costs or reliability risks of current alternatives.

% Section 6: Implementation Strategy and Timeline
\section{\color{primaryblue}Development Plan and Team Coordination}

Our eight-week development timeline follows an iterative approach that prioritizes core functionality while ensuring robust testing and validation throughout the process.

\textbf{Phase 1: Foundation and Architecture (Weeks 1-2)} \\
We begin by establishing the technical foundation through careful dataset selection and core profiler development. This phase focuses on implementing memory-efficient data analysis capabilities and establishing the basic architecture for LLM integration. Key deliverables include a working data profiler capable of handling multi-gigabyte files and initial LLM prompt frameworks.

\textbf{Phase 2: Intelligence and Interface (Weeks 3-4)} \\
The second phase implements the core AI capabilities through systematic LLM prompt engineering and development of the initial user interface prototype. We focus on creating reliable rule generation workflows and establishing the basic validation interface. This phase culminates in a working prototype capable of generating and displaying cleaning rules for sample datasets.

\textbf{Phase 3: Scale and Performance (Weeks 5-6)} \\
Phase three addresses scalability challenges through implementation of the batch processing engine and comprehensive performance optimization. We focus on ensuring the system can handle production-scale datasets efficiently while maintaining rule quality and validation capabilities.

\textbf{Phase 4: Validation and Documentation (Weeks 7-8)} \\
The final phase emphasizes comprehensive testing, performance benchmarking, and documentation creation. We conduct systematic comparisons against existing tools, validate our hallucination mitigation strategies, and create comprehensive user documentation and technical reports.

\textbf{Comprehensive Deliverables Package:}
\begin{itemize}[itemsep=3pt]
\item Production-ready DataMender application with intuitive graphical interface and command-line options
\item Flexible YAML-based configuration system enabling custom rule templates and domain-specific adaptations
\item Professional demonstration video showcasing large-dataset cleaning workflows and performance advantages
\item Rigorous technical report including quantitative performance comparisons against GPT-4, OpenRefine, and Trifacta benchmarks
\item Complete source code repository with comprehensive documentation and installation guides
\end{itemize}

\textbf{Strategic Team Organization:} \\
\textbf{Ahmad Javadi Nezhad} leads project coordination and LLM integration architecture, ensuring coherent system design and effective AI model integration. \\
\textbf{Daniel Bazmandeh} focuses on data profiler optimization and memory management, developing the high-performance foundation for large-scale data processing. \\
\textbf{Iliya Mirzaei} designs and implements the user interface and validation workflows, creating an intuitive experience for rule review and approval. \\
\textbf{Nicholas Tardugno} develops the batch processing engine and data transformation pipelines, ensuring efficient and reliable data cleaning operations. \\
\textbf{Tamali Halder} coordinates testing frameworks, performance metrics collection, and technical documentation, ensuring comprehensive validation and clear communication of results.

Cross-team collaboration occurs throughout all phases, with particular emphasis on dataset curation, user experience testing, and presentation development involving all team members.

% Section 7: Expected Impact and Future Research Directions
\section{\color{primaryblue}Research Contributions and Long-term Vision}

DataMender represents more than a practical tool for data cleaning; it embodies a new paradigm for human-AI collaboration in data preprocessing that has significant implications for the broader field of automated data management.

\textbf{Immediate Research Contributions:} Our work advances several key areas of computer science research. We contribute novel techniques for scaling LLM applications to large datasets through intelligent sampling and incremental processing strategies. Our multi-model validation approach provides a practical framework for mitigating AI hallucination in high-stakes applications. Additionally, our human-in-the-loop validation workflows demonstrate effective patterns for maintaining human agency while leveraging AI automation.

\textbf{Practical Industry Impact:} DataMender addresses a critical pain point experienced across industries, from financial services struggling with transaction data quality to healthcare organizations managing patient record consistency. By dramatically reducing the time and expertise required for data cleaning, we enable organizations to redirect their analytical resources toward higher-value activities like advanced modeling and strategic analysis.

\textbf{Educational and Research Applications:} The open-source nature of DataMender makes advanced data cleaning capabilities accessible to academic researchers and students who previously lacked access to enterprise-grade tools. This democratization of sophisticated data preprocessing capabilities should accelerate research across multiple domains where data quality has been a limiting factor.

\textbf{Future Research Directions:} Success with DataMender opens several promising avenues for future investigation. Automated data schema evolution could extend our rule generation approach to handle structural changes in datasets over time. Intelligent data lineage tracking could leverage LLM capabilities to automatically document data transformation decisions and their business rationale. Predictive data quality monitoring could use historical cleaning patterns to anticipate and prevent data quality issues before they impact downstream processes.

\textbf{Broader Vision:} We envision DataMender as the foundation for a new generation of intelligent data management systems that seamlessly blend human expertise with AI automation. As LLM capabilities continue to advance, the principles and architectures we develop will inform increasingly sophisticated tools for data governance, quality assurance, and automated compliance monitoring.

Our work demonstrates that the future of data science lies not in replacing human expertise with AI automation, but in creating intelligent systems that amplify human capabilities while maintaining the oversight and control necessary for reliable, trustworthy data management.

% References
\section{\color{primaryblue}References}

\begin{enumerate}[itemsep=2pt]
\item Biester, L., et al. (2024). LLMClean: Context-Aware Tabular Data Cleaning via LLM-Generated OFDs. \textit{arXiv preprint arXiv:2404.18681}.

\item Chen, J., et al. (2023). RetClean: Retrieval-Based Data Cleaning Using LLMs and Data Lakes. \textit{arXiv preprint arXiv:2303.16909}.

\item Huang, J., \& Wu, Y. (2024). Cocoon: Semantic Table Profiling Using Large Language Models. \textit{arXiv preprint arXiv:2410.15547}.

\item Li, X., et al. (2024). AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark. \textit{arXiv preprint arXiv:2412.06724}.

\item OpenRefine. (2024). OpenRefine Documentation. Retrieved from https://openrefine.org/docs/

\item Trifacta. (2024). Predictive Transformation Overview. Retrieved from https://docs.trifacta.com/dataprep/en/trifacta-application/concepts/feature-overviews/overview-of-predictive-transformation.html
\end{enumerate}

\end{document}
"""
    
    return latex_content


def main():
    """Generate the LaTeX proposal file and compile to PDF."""
    print("Generating polished DataMender proposal document...")
    
    # Create the document content
    latex_content = create_latex_proposal()
    
    # Get the directory where this script is located (proposal folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_file = os.path.join(script_dir, 'datamender_proposal_polished.tex')
    
    # Write LaTeX file
    with open(tex_file, 'w') as f:
        f.write(latex_content)
    
    print("✅ Generated datamender_proposal_polished.tex")
    
    # Compile to PDF
    print("Compiling to PDF...")
    import subprocess
    try:
        result = subprocess.run(['pdflatex', 'datamender_proposal_polished.tex'], 
                              cwd=script_dir, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Generated datamender_proposal_polished.pdf")
        else:
            print("❌ PDF compilation failed:")
            print(result.stderr)
    except FileNotFoundError:
        print("⚠️  pdflatex not found. Please install LaTeX and run:")
        print("   pdflatex datamender_proposal_polished.tex")
    
    print(f"\n📁 Files created in: {script_dir}")
    print("   - datamender_proposal_polished.tex")
    print("   - datamender_proposal_polished.pdf")


if __name__ == "__main__":
    main()