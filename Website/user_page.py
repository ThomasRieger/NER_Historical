import streamlit as st
import pandas as pd
import io
from utils import extract_named_entities, extract_triples, save_to_database, pull_from_github

def run_user_page():
    if "disclaimer_shown" not in st.session_state:
        st.session_state.disclaimer_shown = False

    if not st.session_state.disclaimer_shown:
        with st.expander("⚠️ Disclaimer", expanded=True):
            st.markdown("""
            **Please Note:** 
            This tool uses automated Named Entity Recognition (NER) and relation extraction.  
            Results may contain inaccuracies or incomplete information.  
            Always **review and verify** before using the extracted data for research, publication, or decision-making.
            This project will collect all inputs and outputs from the user for model improvement and analysis.
                        
            **หมายเหตุ:**
            เครื่องมือนี้ใช้ระบบ Named Entity Recognition (NER) อัตโนมัติและการดึงข้อมูลความสัมพันธ์
            ผลลัพธ์อาจมีความไม่ถูกต้องหรือข้อมูลไม่ครบถ้วน ต้อง **ตรวจสอบและยืนยัน** ทุกครั้งก่อนนำข้อมูลที่ดึงมาใช้เพื่อการวิจัย การตีพิมพ์ หรือการตัดสินใจ
            โครงการนี้จะรวบรวมข้อมูลอินพุตและเอาต์พุตทั้งหมดจากผู้ใช้เพื่อนำไปปรับปรุงและวิเคราะห์แบบจำลอง
            """)
            if st.button("I Understand"):
                st.session_state.disclaimer_shown = True
                st.rerun() # Use st.rerun to reload the page immediately
        st.stop() # Stop execution if disclaimer is not accepted

    st.title("Historical Text Analysis - NER & Triple Extraction")
    
    # Initialize results list in session state if not present
    if "results" not in st.session_state:
        st.session_state.results = []
        
    # Button to sync data from GitHub
    if st.button("Sync/Load Data from GitHub"):
        with st.spinner("Loading data from GitHub..."):
            df = pull_from_github()
            if not df.empty:
                # Convert DataFrame rows to list of dicts
                st.session_state.results = df.to_dict('records')
            else:
                st.session_state.results = []
        st.success("Sync complete!")

    input_mode = st.radio("Select Input Type", ["Manual Text", "Upload CSV"])

    if input_mode == "Manual Text":
        text = st.text_area("Enter your text:", height=150)
        if st.button("Extract"):
            if text.strip():
                with st.spinner("Extracting entities and triples..."):
                    # Call the updated utils functions
                    ner_tuples = extract_named_entities(text)
                    triples = extract_triples(text)
                
                # 1. Generate Raw NER Output (Example: ไต้หวัน:NN|B_COU|B_CLS; พบ:VV|O|I_CLS; ...)
                raw_ner_output = "; ".join([f"{t}:{l}" for t, l in ner_tuples])
                
                # 2. Generate Clean NER Output (Example: ไต้หวัน: B_COU; ...)
                clean_ner_parts = []
                for token, complex_label in ner_tuples:
                    parts = complex_label.split('|')
                    ner_tag = parts[1] if len(parts) > 1 else complex_label
                    
                    if ner_tag and ner_tag != 'O':
                        clean_ner_parts.append(f"{token}: {ner_tag}")

                clean_ner_output = "; ".join(clean_ner_parts)

                # 3. Generate Triples Output (using the new SPO logic)
                triples_output = ", ".join([f"({s}, {p}, {o})" for s, p, o in triples])

                result = {
                    "text": text,
                    "clean_ner": clean_ner_output,
                    "triples": triples_output,
                    "raw_ner": raw_ner_output
                }
                
                # Add to session state (avoid duplicates)
                if result not in st.session_state.results:
                    st.session_state.results.append(result)
                    
                # Save to database
                save_to_database(result)
            else:
                st.warning("Please enter text to extract.")

    else: # Upload CSV
        file = st.file_uploader("Upload your CSV file", type=["csv"])
        
        # Add a dropdown for the user to select encoding
        encoding_options = {
            "utf-8": "UTF-8 (Default)",
            "utf-8-sig": "UTF-8 with BOM (Common from Excel)",
            "cp874": "cp874 (Windows Thai)",
            "tis-620": "tis-620 (Legacy Thai)",
            "latin1": "latin1 (Western European)"
        }
        selected_encoding_label = st.selectbox(
            "Select file encoding (if '???' appears, rewrite the file):", 
            options=encoding_options.values()
        )
        
        # Find the actual encoding key from the selected label
        actual_encoding = [key for key, value in encoding_options.items() if value == selected_encoding_label][0]

        if file:
            try:
                # --- START MODIFICATION ---
                # Read the raw bytes from the uploaded file
                raw_bytes = file.getvalue()
                
                df = None
                try:
                    # Manually decode the bytes using the selected encoding
                    decoded_string = raw_bytes.decode(actual_encoding)
                    # Pass the decoded string to pandas
                    df = pd.read_csv(io.StringIO(decoded_string))
                    
                except UnicodeDecodeError:
                    st.error(f"❌ Read failed. The encoding '{actual_encoding}' is incorrect for this file. Please select another.")
                    return
                except Exception as e:
                    st.error(f"An unexpected error occurred while reading the file: {e}")
                    return
                # --- END MODIFICATION ---

                st.subheader("Preview of Uploaded File")
                # Use st.dataframe for the small preview
                st.dataframe(df.head())

                if "sentence" not in df.columns:
                    st.error("❌ CSV must have a 'sentence' column.")
                    return

                if st.button("Extract from CSV"):
                    
                    new_results = []
                    
                    with st.spinner("Processing all rows..."):
                        progress_bar = st.progress(0, "Starting...")
                        total_rows = len(df)
                        
                        for i, row in df.iterrows():
                            text = str(row["sentence"])
                            if not text.strip():
                                continue
                                
                            ner_tuples = extract_named_entities(text)
                            triples = extract_triples(text)
                            
                            # 1. Raw NER
                            raw_ner_output = "; ".join([f"{t}:{l}" for t, l in ner_tuples])
                            
                            # 2. Clean NER
                            clean_ner_parts = []
                            for token, complex_label in ner_tuples:
                                parts = complex_label.split('|')
                                ner_tag = parts[1] if len(parts) > 1 else complex_label
                                if ner_tag and ner_tag != 'O':
                                    clean_ner_parts.append(f"{token}: {ner_tag}")
                            clean_ner_output = "; ".join(clean_ner_parts)

                            # 3. Triples
                            triples_output = ", ".join([f"({s}, {p}, {o})" for s, p, o in triples])

                            result = {
                                "text": text,
                                "clean_ner": clean_ner_output,
                                "triples": triples_output,
                                "raw_ner": raw_ner_output,
                            }
                            
                            if result not in st.session_state.results:
                                st.session_state.results.append(result)
                                new_results.append(result) # Keep track of new ones to save
                                
                            progress_bar.progress((i + 1) / total_rows, f"Processing row {i+1}/{total_rows}")
                    
                    # Save all new results to database (GitHub)
                    if new_results:
                        st.spinner("Saving new results to database...")
                        # We save one by one, but drop_duplicates will handle it
                        for res in new_results:
                            save_to_database(res)

                    st.success("✅ Extraction complete!")

            except Exception as e:
                st.error(f"An error occurred after reading the CSV: {e}")

    # --- Display Results Table (runs for both modes) ---
    if st.session_state.results:
        st.subheader(f"All Extracted Entries ({len(st.session_state.results)} total)")
        
        # Create DataFrame from session state
        result_df = pd.DataFrame(st.session_state.results)
        
        # Reorder and rename columns for better display
        display_columns = {
            "text": "Text",
            "clean_ner": "NER Entities",
            "triples": "Extracted Triples (SPO)",
            "raw_ner": "Raw Output"
        }
        
        # Ensure all expected columns exist, even if empty
        for col_key in display_columns.keys():
            if col_key not in result_df.columns:
                result_df[col_key] = ""
                
        # Filter to only show desired columns in the right order
        display_df = result_df[list(display_columns.keys())].rename(columns=display_columns)
        
        # Use st.table for basic, full-content text wrapping
        st.table(display_df)

        # Download button
        csv_buffer = io.BytesIO()
        # Save the original data (not the renamed columns)
        result_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        csv_buffer.seek(0)
        st.download_button(
            "Download All Results as CSV", 
            csv_buffer, 
            "results.csv", 
            "text/csv",
            key="download_csv"
        )

