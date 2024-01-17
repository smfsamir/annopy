Example

```
annotation_frame = annotation_frame.with_columns(pl.col('context').list.to_struct()).unnest("context")\
        .rename({'field_0': 'context_0', 'field_1': 'context_1'})

annotation_frame = load_save_if_nexists(annotation_frame, f"{ANNOTATION_SAVE_PATH}/annotation_{today_str}.csv")
def ask_question(fact_row):
        context_str = f"- {fact_row['context_0']}\n" + f"- {fact_row['context_1']}\n" # NOTE: could this be empty?
        facts = context_str + f"- {fact_row['fact']}"
        tgt_lang = 'French' if fact_row['src_language'] == 'en' else 'English'
        question = f"\n\nConsider the following fact(s) about {fact_row['person_name']}:\n\n{facts}\n\nIs the final fact present in the {tgt_lang} Wikipedia article about {fact_row['person_name']}? ( Yes / No ): "
        return question

annotated_frame = annotate_frame(
        annotation_frame, # frame containing data to annotate 
        num_samples=10, # number of samples to annotate in one setting
        annotation_columns=['fact_in_tgt'], # column to store the annotation in
        question_fns=[ask_question],
        answer_validate_fn=[lambda answer: answer.lower() in ['yes', 'no']],
)
annotated_frame.write_csv(f"{ANNOTATION_SAVE_PATH}/annotation_{today_str}.csv")
return
```
