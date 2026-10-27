# Stack Overflow CQA Dataset (10,000 questions)

Dataset matching the Stack Exchange Data Explorer-style query below, exported as CSV for local ML / research use (question routing, quality assessment, etc.).

## File

- `stackoverflow_questions_2023plus.csv` — **10,000** Stack Overflow questions

## Columns

| Column | Description |
|---|---|
| QuestionId | Post Id |
| Title | Question title |
| Body | Question body (HTML) |
| Tags | Pipe-delimited tags, e.g. `\|python\|pandas\|` |
| Score | Net votes |
| ViewCount | Views |
| AnswerCount | Number of answers |
| CommentCount | Number of comments |
| FavoriteCount | Favorites / bookmarks |
| CreationDate | UTC `YYYY-MM-DD HH:MM:SS` |
| LastEditDate | UTC (empty if never edited) |
| AcceptedAnswerId | Accepted answer Id (empty if none) |
| UserId | Asker user Id |
| UserReputation | Asker reputation at fetch time |

## Query equivalent

```sql
SELECT TOP 10000
    p.Id AS QuestionId,
    p.Title,
    p.Body,
    p.Tags,
    p.Score,
    p.ViewCount,
    p.AnswerCount,
    p.CommentCount,
    p.FavoriteCount,
    p.CreationDate,
    p.LastEditDate,
    p.AcceptedAnswerId,
    u.Id AS UserId,
    u.Reputation AS UserReputation
FROM Posts p
LEFT JOIN Users u ON p.OwnerUserId = u.Id
WHERE p.PostTypeId = 1
  AND p.CreationDate >= '2023-01-01'
  AND p.Score IS NOT NULL
ORDER BY p.CreationDate DESC;
```

## Source

Fetched via the official [Stack Exchange API](https://api.stackexchange.com/) (`/2.3/questions`, site=`stackoverflow`) with polite pagination — **not** by scraping HTML (avoids rate-limit blocks from Data Explorer / site scraping).

Content is licensed under [CC BY-SA](https://stackoverflow.com/help/licensing) (Stack Overflow / Stack Exchange).

## Quick load

```python
import pandas as pd
df = pd.read_csv("stackoverflow_questions_2023plus.csv")
print(df.shape)
print(df.head())
```
