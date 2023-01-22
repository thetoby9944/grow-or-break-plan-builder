import itertools

import PIL.Image
import streamlit as st
import numpy as np
import pandas as pd
import re

from streamlit_sortables import sort_items
from streamlit_tags import st_tags
from st_aggrid import AgGrid, ColumnsAutoSizeMode, AgGridTheme, GridOptionsBuilder

from utils import pad_dict_list, make_grid

ss = st.session_state
main_content_table_theme = AgGridTheme.ALPINE

st.image("180.png")

"# **Grow or Break** Plan Builder"
"# "
"# "

# "## Start with Sessions"


"## ① Exercises that go together (Sessions)"

name_sessions_container = st.container()
groups_container = st.container()
sessions_container = st.container()
add_ex_container = st.container()
"---"
session_repeats = st.container()
session_sorting = st.container()

"## ② Whats your training style (Sets & Reps)"

all_set_styles_container = st.container()
add_set_style_container = st.container()


"## ③ Progressive overload! (Waves & Periodization)"

all_set_progressions = st.container()
add_progression_container = st.container()


"## ④ Lets build the program! (Blocks and Exercise Types)"

default_set_style_container = st.container()
conditional_set_style_container = st.container()
add_rules_container = st.container()


exercises = pd.read_csv("exercises.csv", sep="\t")

"## ⑤ Export"

export_container = st.container()

if "df" not in ss:
    ss["df"] = exercises



n_col = ss.df.shape[1]  # col count
rw = -1

with add_ex_container.expander("Can't find your favourite exercise? Add it!" ):
    with st.form(key="add form", clear_on_submit=True):
        cols = st.columns(n_col - 1)
        df = ss.df

        raw_data = {
            "Exercise": cols[0].text_input("Exercise Name"),
            "Muscle": cols[1].selectbox("Muscle", df["Muscle"].unique()),
            "Exercise Type": cols[2].selectbox("Exercise Type", df["Exercise Type"].unique())
        }

        # you can insert code for a list comprehension here to change the data
        # values into integer / float, if required

        if st.form_submit_button("Add"):
            if raw_data["Exercise"] in ss.df.Exercise:
                st.warning("Exercise name already exists.")
            if raw_data["Exercise"] == "":
                st.warning("Please specify a name")
            else:
                df = ss.df
                raw_data["Muscle Group"] = df[df["Muscle"] == raw_data["Muscle"]].at[0, "Muscle Group"]
                rw = ss.df.shape[0] + 1
                ss.df.loc[rw] = raw_data
                st.info("Added exercise")

with st.sidebar.expander("All Exercises"):
    df = ss.df
    AgGrid(df.reindex(columns=np.roll(df.columns, 1)), height=250, theme="material")

with groups_container.expander("Exercise groups"):
    st.info(
        "**Exercise Groups** \n\n You can add your own exercise groups here. "
        "Now you can add your group to your sessions"
    )

    defaults = {
        "Cardio": [
            "Incline Walk",
            "Stairmaster"
        ]
    }

    placeholder = st.container()

    group_names = list(defaults.keys())

    "---"
    group_names = st_tags(
        label="Add more Exercise Groups. Careful, this will reset your sessions.",
        value=group_names,
        suggestions=["Arms", "Core", "Accessories"],
        text="Enter new group name here and press ⏎"
    )

    if not len(group_names):
        st.warning("Create at least one exercise group please")
        st.stop()

    groups = {
        group:
            placeholder.multiselect(
                group,
                options=ss.df.Exercise,
                default=defaults[group] if group in defaults else None,
                format_func=lambda x: x,
                help="Start writing in the group to search for exercises."
            )
        for group in group_names
    }

with st.sidebar.expander("My groups"):
    AgGrid(pd.DataFrame(pad_dict_list(groups, "")), theme="material")

with name_sessions_container:
    default_sessions = {
        "Day 1": [
            "Bench Press Barbell",
            "Hack Squats",
            "Overhead Shoulder Press Barbell",
            "Pull-Ups",
            "Cardio"
        ],
        "Day 2": [
            "Conventional Deadlifts",
            "Bench Press Dumbbell",
            "Pull-Ups",
            "Barbell Row Standing",
            "Cardio"

        ],
        "Day 3": [
            "Hack Squats",
            "Conventional Deadlifts",
            "Barbell Row Standing",
            "Overhead Shoulder Press Barbell",
            "Cardio"
        ],

    }

    sessions = list(default_sessions.keys())

    sessions = st_tags(
        label="Name your training sessions",
        value=sessions,
        suggestions="""
        WEEK A - Day 1  
    WEEK A - Day 2  
    WEEK A - Day 3  
    WEEK B - Day 1  
    WEEK B - Day 2  
    WEEK B - Day 3  """.split("\n"),
        text="Enter new session name here and press ⏎",
        key="sttags"
        # help=sessions_help
    )

    if not len(sessions):
        st.warning("Name at least one session please")
        st.stop()

    sessions_list = sessions

with sessions_container:
    group_names = [group for group in groups.keys()]
    options = group_names + list(ss.df.Exercise.tolist())
    sessions_with_elements = {
        session: st.multiselect(
            session,
            options=options,
            default=
            (value for value in default_sessions[session] if value in options)
            if session in default_sessions
            else []
        )
        for i, session in enumerate(sessions_list)

    }

    sessions_with_exercises = {
        session: [
            exercise
            for session_element in session_elements
            for exercise in (groups[session_element] if session_element in groups else [session_element])
        ]
        for session, session_elements in sessions_with_elements.items()

    }

with st.sidebar.expander("My sessions"):
    table = pd.DataFrame(pad_dict_list(sessions_with_exercises, ""))
    table.index += 1
    AgGrid(table, theme="material")

with session_repeats:
    n_blocks = int(st.number_input("Repeat times", 0, max_value=52, value=16))

    f"Congrats. Your program will have {int(n_blocks)} blocks. Blocks will look like this" \
    f" (Drag and drop the exercises to sort)"

with session_sorting:
    # if st.checkbox("Enable sorting"):
    session = sessions * 2
    grid = make_grid(len(sessions) // 3 + 1, 3)
    for i, (session, exercises) in enumerate(sessions_with_exercises.items()):
        with grid[i // 3][i % 3]:
            st.write(session)
            sessions_with_exercises[session] = sort_items(exercises, direction="vertical")

set_styles = pd.read_csv("set_styles.csv", sep="\t")

if "df_sets" not in ss:
    ss["df_sets"] = set_styles

n_col = ss.df_sets.shape[1]  # col count
rw = -1


def is_set_style(input_text):
    pattern = re.compile(r"^[+-]?\d+\s*(/\s*[+-]?\d+\s*)*$", re.IGNORECASE)
    res = pattern.match(input_text)
    if not res:
        st.warning("Set Style should look like this \n\n    12 / 18 / 10 \n\n or \n\n    5 / 5 / 5 / 5 \n\n"
                   "Type negative values to specify number of reps left in the tank . \n\n Type zero to specify an "
                   "AMRAP (as many reps as possible) set.")
    return res


def is_weight_adjustment(input_text):
    pattern = re.compile(
        r"^("
        r"([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[eE]([+-]?\d+))?%\s*"
        r"(/\s*([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[eE]([+-]?\d+))?%\s*)*"
        r")?$",
        re.IGNORECASE)
    res = pattern.match(input_text)
    if not res:
        st.warning("Weight adjustment should look like this: `-12.5% / -5%` or `-12.5% / -5% / 0% / -5%`")
    return res


with add_set_style_container.expander("Missing your favourite set style? Add it!"):
    with st.form(key="add progression_form", clear_on_submit=False):
        cols = st.columns(n_col)

        raw_data = {
            "Name": cols[0].text_input("Name", placeholder="Set Style Name"),
            "Reps": cols[1].text_input("Reps", value="5 / 3 / 1", placeholder="12 / 10 / 8 / 10"),
            "Warmup Weight Adjustment": cols[2].text_input("Weight Adjustment per set", placeholder="-12% / -7.5%")
        }

        # you can insert code for a list comprehension here to change the data
        # values into integer / float, if required

        if st.form_submit_button("Add"):
            if raw_data["Name"] in ss.df_sets.Name.values:
                st.warning("Set style already exists!")
            elif not raw_data["Name"]:
                st.warning("Please name the set style!")
            elif is_set_style(raw_data["Reps"]) and is_weight_adjustment(raw_data["Warmup Weight Adjustment"]):
                df = ss.df_sets
                rw = ss.df_sets.shape[0] + 1
                ss.df_sets.loc[rw] = raw_data
                st.info("Added set style")


if "df_progressions" not in ss:
    ss["df_progressions"] = pd.read_csv("progressions.csv")


with all_set_progressions:
    st.subheader("Set Progressions")
    df = ss.df_progressions
    AgGrid(df, height=250, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS, theme=main_content_table_theme)

with add_progression_container.expander("Want a custom progression wave, i.e. weight adjustments per block? Add it!"):
    with st.form(key="add set_form", clear_on_submit=False):
        cols = st.columns(2)

        raw_data = {
            "Name": cols[0].text_input("Progression Name", placeholder="E.g. accessory progression"),
            "Progression": cols[1].text_input(
                "Progression (weight adjustment for each block)",
                value="60% / 62% / 65% / 62% / 67% / 70% / 67% / 70% / 72% / 70% / 72% / 75% / 72% / 75% / 77% / 75%",
                placeholder="70% / 80% / 90% / 100%"
            ),

        }

        if st.form_submit_button("Add"):
            if raw_data["Name"] in ss.df_progressions.Name.values:
                st.warning("Set porgression already exists!")
            elif not raw_data["Name"]:
                st.warning("Please name the set style!")
            elif is_weight_adjustment(raw_data["Progression"]):
                df = ss.df_progressions
                rw = ss.df_progressions.shape[0] + 1
                ss.df_progressions.loc[rw] = raw_data
                st.info("Added progression")

with all_set_styles_container:
    st.subheader("Set Styles")
    df = ss.df_sets
    AgGrid(df, height=250, theme=main_content_table_theme, fit_columns_on_grid_load=True)

# "## What's your routine bro?"
# with default_set_style_container:
#    default_set_style = st.selectbox("Default set style", options=ss.df_sets.Name, index=6)

with conditional_set_style_container:
    st.write("Here's an example how you would configure the Juggernaut program. "
             "See https://liftvault.com/programs/strength/juggernaut-method-base-template-spreadsheet/"
             " for source and details.")
    st.image(PIL.Image.open("./juggernaut_programm.jpeg"))
    sessions_df = pd.DataFrame(
        {
            "Session": sessions,
        }
    )
    blocks_df = pd.DataFrame(
        {
            "Block": list([f"Block {i + 1}" for i in range(n_blocks)])
        }
    )
    session_exercise_df = pd.DataFrame(sessions_with_exercises).add_prefix("Exercise of ")
    all_exercises_by_day = pd.DataFrame({
        "Exercise by Day": [
            session + " " + exercise
            for session, exercises in sessions_with_exercises.items()
            for exercise in exercises
        ]}
    )
    all_exercises_in_plan = pd.DataFrame({
        "Planned Exercise": np.unique([
            exercise
            for session, exercises in sessions_with_exercises.items()
            for exercise in exercises
        ])}
    )

    df = pd.concat([
        ss.df,
        sessions_df,
        # session_exercise_df,
        blocks_df,
        # all_exercises_by_day,
        # all_exercises_in_plan
    ], axis=1)

    df_2 = df.add_suffix(" is").add_prefix("If ")

    juggernaut = """10's Accumulation
10's Intensification
10's Realization
Deload
8's Accumulation
8's Intensification
8's Realization
Deload
5's Accumulation
5's Intensification
5's Realization
Deload
3's Accumulation
3's Intensification
3's Realization
Deload"""

    rule_df_temp_01 = pd.DataFrame(columns=df_2.columns).drop("If Block is", axis=1)

    rule_df_temp_02 = pd.DataFrame({
        "If Block is": blocks_df.Block,
    })

    rule_df_temp_1 = pd.concat([rule_df_temp_02, rule_df_temp_01], axis=1)

    rule_df_temp_2 = pd.DataFrame({
        # "Rule Number" : list(range(n_blocks)),
        "Set Style": juggernaut.split("\n"),
    })

    if "rule_df" not in ss:
        ss["rule_df"] = pd.read_csv("./rules.csv")#pd.concat([rule_df_temp_1, rule_df_temp_2], axis=1).dropna(axis=1, how="all").fillna("Any")


    AgGrid(ss.rule_df, height=250, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS, theme=main_content_table_theme)

    if st.button("Reset"):
        ss.rule_df = pd.DataFrame()


with add_rules_container.expander("Add Rules", expanded=True):
    if "n_rules" not in ss:
        ss["n_rules"] = 1

    rule_columns = st.multiselect("Conditions", rule_df_temp_1.columns, default=["If Exercise Type is"], key="basedon",
                                  kwargs=dict(label_visibility="collapsed"))

    conditions = {}
    concatenator = ""
    for i, column in enumerate(rule_columns):

        rules = [
            (concatenator, column, selection)
            for selection in st.multiselect(column, list(df_2[column].dropna().unique()))
        ]

        conditions[column] = rules

        if i < len(rule_columns) - 1:
            concatenator = "and "

    from itertools import product

    conditions = list(product(*conditions.values()))

    if np.asarray(conditions).ndim == 2:
        conditions = [[condition] for condition in conditions]

    if len(conditions):
        st.write("---")
        st.write("Rules")


    # op1 = "for all matches"
    # op2 = "for each match separately"

    select_style_radio = True  # op1
    if len(conditions) > 1:
        select_style_radio = st.checkbox("Same rules for all conditions",
                                         True)  # st.radio("Select set style", [op1, op2])

    if len(rule_columns) < 1 or len(conditions) < 1:
        set_styles_by_condition = {}

    elif select_style_radio:

        all_match_set_style = st.selectbox(
            "Set Style",
            ss.df_sets.Name
        )

        all_match_progression = st.selectbox(
            "Block Progression",
            ss.df_progressions.Name
        )

        #all_match_ignore = st.checkbox("Ignore Weights", False)

        set_styles_by_condition = [
            {
                **{
                    column: selection
                    for (concatenator, column, selection) in condition
                },
                **{
                    "Set Style": all_match_set_style
                },
                **{
                    "Block Progression": all_match_progression
                },

            }
            for condition in conditions
        ]

    else:

        set_styles_by_condition = [
            {
                **{
                    column: selection
                    for (concatenator, column, selection) in condition
                },
                **{
                    "Set Style": st.selectbox(
                        " ".join(np.asarray(condition).flatten())+ "Set Style",
                        ss.df_sets.Name
                    )
                },
                **{
                    "Block Progression": st.selectbox(
                        " ".join(np.asarray(condition).flatten()) + "Block Progression",
                        ss.df_progressions.Name
                    )
                },

            }
            for condition in conditions
        ]

    if len(set_styles_by_condition) > 0:
        new_rules_df = pd.DataFrame(set_styles_by_condition)
        AgGrid(new_rules_df, theme="material")
        if st.button("Add Rules"):
            ss.rule_df = pd.concat([ss.rule_df, new_rules_df], axis=0)
            ss.rule_df.reset_index(inplace=True, drop=True)
            ss.rule_df = ss.rule_df[
                [col for col in ss.rule_df.columns if col.startswith("If")] +
                [col for col in ss.rule_df.columns if not col.startswith("If")]
            ]
            ss.rule_df = ss.rule_df.replace("Ignore", None)
            st.info("Added rules")


with export_container.expander("Enter your 1RMs before generating your plan!").form("Enter your 1RMs"):
    st.write("View this 1RM calculator if you don't know your 1RMs exactly.")
    st.image(PIL.Image.open("./1RM_calculator.png"))
    one_rms = {}

    plan_structure = {}
    for block_number, block in enumerate(blocks_df["Block"]):
        plan_structure[block] = {}
        for session, exercises in sessions_with_exercises.items():

            for exercise in exercises:
                if exercise not in one_rms:
                    one_rms[exercise] = st.number_input(f"1RM for {exercise}", 100)

                exercise_row = ss.df.loc[ss.df["Exercise"] == exercise].iloc[0]

                match = {
                    "If Block is": block,
                    "If Exercise is": exercise,
                    "If Exercise Type is": exercise_row["Exercise Type"],
                    "If Muscle is": exercise_row["Muscle"],
                    "If Muscle Group is": exercise_row["Muscle Group"],
                    "If Session is": session
                }
                if "Set Style" in ss.rule_df:
                    current_set_style_rules = [
                        row for index, row in ss.rule_df[ss.rule_df['Set Style'].notna()].to_dict("index").items()
                        if all(
                            col not in row or row[col] == val or row[col] is np.nan
                            for col, val in match.items()
                        )
                    ]
                else:
                    current_set_style_rules = []

                if "Block Progression" in ss.rule_df:
                    current_progression_rules = [
                        row for index, row in ss.rule_df[ss.rule_df['Block Progression'].notna()].to_dict("index").items()
                        if all(
                            col not in row or row[col] == val or row[col] is np.nan
                            for col, val in match.items()
                        )
                    ]
                else:
                    current_progression_rules = []

                if len(current_set_style_rules) > 0:
                    current_set_style = current_set_style_rules[-1]["Set Style"]

                else:
                    current_set_style = "3x8"

                if len(current_progression_rules) > 0:
                    current_progression = current_progression_rules[-1]["Block Progression"]
                    progression_row = ss.df_progressions.loc[ss.df_progressions["Name"] == current_progression].iloc[0]
                    current_progressions_for_all_blocks = [
                        float(weight.replace("%", "e-2").replace(" ", ""))
                        for weight in progression_row["Progression"].split("/")
                    ]
                    if block_number < len(current_progressions_for_all_blocks):
                        progression = current_progressions_for_all_blocks[block_number]
                    else:
                        progression = 1
                else:
                    progression = 1

                if current_set_style in ss.df_sets["Name"].tolist():
                    set_style_row = ss.df_sets.loc[ss.df_sets["Name"] == current_set_style].iloc[0]
                else:
                    st.warning(f"{current_set_style} not in set styles")

                reps = [
                    float(rep.strip())
                    for rep in set_style_row['Reps'].split("/")
                ] if set_style_row['Reps'] is not np.nan else []

                weights_percentage =  [
                    float(weight.replace("%", "e-2").replace(" ", ""))
                    for weight in set_style_row['Warmup Weight Adjustment'].split("/")
                ] if set_style_row["Warmup Weight Adjustment"] is not np.nan else []

                current_set = " ".join([
                    f"{int(rep_count)}"
                    f"x{round(one_rms[exercise]*(progression+weight_percentage))}"
                    #f"({(progression+weight_percentage)*100}% of {one_rms[exercise]})"
                    for i, (rep_count, weight_percentage)
                    in enumerate(list(itertools.zip_longest(reps, weights_percentage, fillvalue=0)))
                ])

                if len(reps) > 0 and exercise not in one_rms and current_set_style != "Empty":
                    one_rms[exercise] = st.number_input(f"1RM for {exercise}", 100)

                plan_structure[block][session + " "+ exercise] = current_set

    if st.form_submit_button("Submit 1RMs"):
        pass

with export_container:
    if st.button("Generate Plan"):
        result_plan = pd.DataFrame.from_dict({

                (i,j): plan_structure[i][j]
                for i in plan_structure.keys()
                for j in plan_structure[i].keys()
                #for k in plan_structure[i][j].keys()

            },
            orient="index"
        )#.reset_index(level=[0,1]).pivot(index="level_1", columns=["level_0"], values=["Reps", "Weight Percentage"])
        result_plan = pd.DataFrame.from_dict(plan_structure)
        #result_plan =pd.json_normalize(plan_structure, meta=['Blocks','Sessions', "Sets"])#, record_path='Sets')
        result_plan["Exercise"] = result_plan.index
        result_plan = result_plan.reindex(columns=np.roll(result_plan.columns, 1))

        gb = GridOptionsBuilder.from_dataframe(result_plan)
        gb.configure_default_column(
            resizable= True,
            autoHeight=True,

            #wrapText=True,
            width=150,
            cellStyle={
                #'white-space': 'pre',
                'overflow-wrap': "anywhere",
                "overflow": "visible",
        "text-overflow": "unset",
        "white-space": "normal",
            },
        )
        AgGrid(result_plan, gridOptions=gb.build(), height=400, theme=main_content_table_theme)# columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)

        st.info("Right click the plan to export!")