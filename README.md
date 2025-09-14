# Cevovod CSVSI za bodoči vtičnik dcat-custom-deep

Ta repozitorij vsebuje kodo in Jupyter zvezke, uporabljene v okviru magistrske naloge  
**"Avtomatsko generiranje semantičnih podatkovnih shem za nove vire v portalih odprtih podatkov"**.

## Struktura repozitorija

- `*.py` – skripte za posamezne korake cevovoda (generiranje opisov, pretvorbe formatov, generiranje CSVW).
- `notebook/` – vsebuje zvezek Jpyter za testiranje ujemanja in analizo rezultatov za ujemnje na primerih OAEI in OPSI
- `data/` – vhodni primeri CSV datotek in ontologij.
- `results/` – izhodni JSON in TTL rezultati eksperimentov.
- `notebook_*/` - vsebuje preizkuse ujemanja na različnih virih s poratala OPSI iz domene podatkov o javnih parkiriščih za ujemanje ontologij in analizo rezultatov. |

## Skripte

| Datoteka | Opis |
|----------|------|
| `column_rag_generation_opsi_openai.py` | Generira opise stolpcev iz CSV datotek z uporabo LLM (OpenAI 4.1). |
| `source_csvw_json_to_ontology_format.py` | Pretvori generiran CSVW JSON v obliko izvorne ontologije primerno za ujemanje. |
| `csvvw_to_json_converter.py` | Pretvori TTL zapis ontologije v JSON obliko primerno za ujemanje. |
| `genriraj_csvw_ttl.py` | Iz JSON zapisov generira končno `.ttl` datoteko po standardu CSVW z uporabo RDFLib. |

## Zahteve

- Python 3.10+
- [requirements.txt](requirements.txt) (nameščanje z `pip install -r requirements.txt`)
- Za grajenje baze FAISS pri ujemnju je priporočena uporaba **GPU pospeševanja** (CUDA) zaradi časovne zahtevnosti

