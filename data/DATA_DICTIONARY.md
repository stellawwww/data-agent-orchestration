# Data dictionary

What an analyst at this (fictional) healthcare booking marketplace would know about its tables.
Patients search for a specialty in a market, see providers with open slots, open a provider
profile, and book. Providers publish appointment slots.

Coverage: 2024-02-01 to 2026-01-31, daily. 5 markets, 6 specialties, 3 platforms.

## daily_funnel
One row per date × market × specialty × platform. Source: app event tracking.

| Column | Meaning |
|---|---|
| `date` | Calendar day (local market time) |
| `market` | NYC, Chicago, Houston, Phoenix, Atlanta |
| `specialty` | Primary care, Urgent care, Flu shot, Dermatology, Dentist, Therapy |
| `platform` | iOS, Android, Web |
| `searches` | Patient searches for this specialty in this market |
| `searches_with_availability` | Searches that returned at least one provider with an open slot |
| `profile_views` | Clicks from search results into a provider profile |
| `bookings` | Appointments booked, as recorded by the app's booking event |

## daily_supply
One row per date × market × specialty. Source: provider scheduling system (server side, not app tracking).

| Column | Meaning |
|---|---|
| `date`, `market`, `specialty` | As above |
| `active_providers` | Providers with any open schedule that day |
| `slots_offered` | Bookable appointment slots published that day |
| `slots_booked` | Slots filled that day, as recorded by the scheduling system |

`slots_booked` and the sum of `daily_funnel.bookings` across platforms measure the same bookings
through two independent systems.

## flu_activity
One row per week × region. Source: CDC ILINet (real public data, via the CMU Delphi Epidata API).

| Column | Meaning |
|---|---|
| `week_start` | Sunday that starts the CDC epiweek |
| `epiweek` | CDC epiweek as YYYYWW |
| `region` | CDC region code; see `markets.csv` for which market uses which |
| `wili` | Weighted share of outpatient visits for influenza-like illness, as a decimal (0.0832 = 8.32%) |

## markets
| Column | Meaning |
|---|---|
| `market` | Market name |
| `flu_region` | Region in `flu_activity` for this market (NYC uses HHS Region 2, New York + New Jersey) |

## calendar
| Column | Meaning |
|---|---|
| `date`, `day_of_week` | — |
| `is_holiday` | Major US holiday when most offices close |
| `holiday_name` | Holiday or holiday-adjacent day (e.g. Day after Thanksgiving), blank otherwise |
| `is_holiday_week` | Thanksgiving Wednesday to Sunday, and Dec 22 to Jan 2 |

## change_log
Product launches, app releases and marketing campaigns, as logged by the teams that ran them.
Not everything that happens gets logged.

| Column | Meaning |
|---|---|
| `event_date` | Launch or start date |
| `event_type` | `product_launch`, `app_release`, `marketing` |
| `scope` | Markets, specialties or platforms affected |
| `description` | One-line description |

## Standard metrics
```
availability_rate = searches_with_availability / searches
click_rate        = profile_views / searches_with_availability
booking_cvr       = bookings / profile_views
bookings          = searches × availability_rate × click_rate × booking_cvr

fill_rate         = slots_booked / slots_offered
```
