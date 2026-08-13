# Ground Truth Label Format

Each video requires a matching `.csv` label file in the same directory.

## File naming

Video:  `videos/fall_no_recovery/test1.mp4`
Labels: `videos/fall_no_recovery/test1.csv`

## CSV format

```
frame,event
0,normal
45,fall_start
52,fall_confirmed
180,emergency
360,end
```

## Event types

| Event | Meaning |
|---|---|
| `normal` | Person is upright, walking, standing |
| `sitting` | Person is intentionally sitting |
| `lying` | Person is intentionally lying down (not a fall) |
| `bending` | Person is bending or picking something up |
| `exercise` | Person is doing floor exercise / yoga |
| `fall_start` | Frame where fall motion begins |
| `fall_confirmed` | Frame where person is clearly down |
| `emergency` | Frame where person has been down long enough to be an emergency |
| `recovery` | Frame where person begins to get up |
| `recovered` | Frame where person is fully upright again |
| `end` | Last relevant frame |

## Rules

- `frame` is 0-indexed
- Each row marks the START of that event
- A video may have multiple events (e.g. person sits, then falls)
- If no fall occurs, only `normal` and `end` are needed
- Frames between events inherit the previous event label

## Example: fall with recovery

```
frame,event
0,normal
60,fall_start
70,fall_confirmed
150,recovery
180,recovered
200,normal
300,end
```

## Example: intentional lying (should NOT trigger emergency)

```
frame,event
0,normal
30,lying
200,normal
300,end
```
