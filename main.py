from streamlit_sortables import sort_items
from streamlit_tags import st_tags
from st_aggrid import AgGrid, ColumnsAutoSizeMode, AgGridTheme, GridOptionsBuilder

import datetime
import io
import itertools

import PIL.Image
import streamlit as st
import numpy as np
import pandas as pd
import re
from time import sleep


@st.cache
def load_app():
    with st.spinner("Loading"):
        sleep(3)


# Constants

ss = st.session_state
main_content_table_theme = AgGridTheme.ALPINE.value



# UI Order

with st.columns(5)[0]:
    st.image("assets/gob_logo.png", use_column_width=False, width=150)

"# **Grow or Break** Plan Builder"
"# "
"# "

# "## Start with Sessions"
load_col, save_col = [st.sidebar.container()]*2 #st.sidebar.columns(2)
st.sidebar.write("---")
load_container = load_col.container()
save_container = save_col.container()


with load_container:
    st.subheader("Load")
    st.warning("Loading a template or reloading the page will reset your changes.")
    default_plan_name = st.selectbox(
        "Select a template",
        ["Empty", "Juggernaut", "Upload Custom Plan"],
        on_change=lambda: ss.clear()
    )
    if default_plan_name == "Upload Custom Plan":
        upload = st.file_uploader("Upload a savefile", type="gob", on_change=lambda: ss.clear())
        if upload is not None:
            file_ref = upload
        else:
            st.info("Upload a previously saved .gob template to continue.")
            st.stop()
    else:
        file_ref = default_plan_name + " Template.xlsx"




# Load Defaults

if "id" not in ss:
    ss["id"] = datetime.datetime.now().isoformat()

if "enable_groups" not in ss:
    ss["enable_groups"] = False

if "set_format" not in ss:
    ss["set_format"] = "{reps}x{weight}"
    ss["set_format_line_2"] = ""
    ss["set_format_delimiter"] = " "
if "df" not in ss:
    ss["df"] = pd.read_excel(file_ref, "Exercises")

if "df_sets" not in ss:
    ss["df_sets"] = pd.read_excel(file_ref, "Set Styles")

if "df_progressions" not in ss:
    ss["df_progressions"] = pd.read_excel(file_ref, "Progressions")

if "rule_df" not in ss:
    ss["rule_df"] = pd.read_excel(file_ref, "Rules")

if "df_sessions" not in ss:
    ss["df_sessions"] = pd.read_excel(file_ref, "Sessions")
    ss["default_sessions"] = {
        session: exercises.dropna().to_list()
        for session, exercises
        in ss.df_sessions.to_dict("series").items()
    }

if "df_blocks" not in ss:
    ss["df_blocks"] = pd.read_excel(file_ref, "Blocks")
    ss["default_blocks"] = ss.df_blocks["Number of Blocks"].loc[0]

if "sort_key" not in ss:
    ss["sort_key"] = datetime.datetime.now().isoformat()



"## ① Training Sessions"

st.sidebar.header("Exercises")
add_ex_container = st.sidebar.container()  # expander("Can't find your favourite exercise? Add it!" )
all_exercises_container = st.sidebar.expander("View Exercises")

name_sessions_container = st.container()

with name_sessions_container:
    sessions = list(ss.default_sessions.keys())

    sessions = st_tags(
        label="",
        value=sessions,
        suggestions="""
        WEEK A - Day 1  
    WEEK A - Day 2  
    WEEK A - Day 3  
    WEEK B - Day 1  
    WEEK B - Day 2  
    WEEK B - Day 3  """.split("\n"),
        text="Enter new session name here and press ⏎",
        key="sttags"+ss.id
        # help=sessions_help
    )

    if not len(sessions):
        st.warning("Name at least one session, e.g. 'Day 1' or 'Upper Body' or ...")
        st.stop()

    sessions_list = sessions



# all_groups_container = st.sidebar.expander("My groups")
groups_container = st.expander("Exercise groups")

sessions_container = st.container()

"---"
"## ② Number of training blocks"

session_repeats = st.container()
session_sorting = st.container()
all_sessions_container = st.sidebar.expander("View Sessions")

st.sidebar.header("Set Styles")
add_set_style_container = st.sidebar.container()  # expander("Missing your favourite set style? Add it!")
all_set_styles_container = st.sidebar.expander("View Set Styles")

# "## ④⑤ Progressive overload! (Waves & Periodization)"

st.sidebar.header("Progression Waves")
add_progression_container = st.sidebar.container()  # expander("Want a custom progression wave? Add it!")
all_set_progressions = st.sidebar.expander("View Progression Waves")


"---"

"## ③ Set the Program Rules!"

# default_set_style_container = st.container()
conditional_set_style_container = st.container()
add_rules_container = st.expander("Add Rules", expanded=True)

"---"

"## ④ Export Plan"

one_rm_container = st.expander("Enter your 1RMs before generating your plan!")
export_plan_container = st.container()

st.sidebar.header("Settings")
settings_container = st.sidebar.container()



# Utils

def pad_dict_list(dict_list_orig: dict, padel):
    dict_list = {key: [i for i in value]for key, value in dict_list_orig.items()}
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if  ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list


def make_grid(rows, cols):
    return {
        row :
        st.columns(cols)
        for row in range(rows)
    }


def flatten(item, sequences=(tuple, list, set)):
    yield from map(flatten, item) if isinstance(item, sequences) else [item]


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



# Program Start

with add_ex_container:
    n_col = ss.df.shape[1]  # col count
    rw = -1

    with st.form(key="add form", clear_on_submit=True):
        cols = [st.container(), st.container(), st.container()]  # st.columns(3)
        df = ss.df

        raw_data = {
            "Exercise": cols[0].text_input("Add Exercise", placeholder="Exercise Name"),
            "Muscle": cols[1].selectbox("Muscle", df["Muscle"].unique()),
            "Exercise Type": cols[2].selectbox("Exercise Type", df["Exercise Type"].unique())
        }

        if st.form_submit_button("Add"):

            if raw_data["Exercise"] == "":
                st.warning("Please specify a name")
            else:
                df = ss.df
                raw_data["Muscle Group"] = df[df["Muscle"] == raw_data["Muscle"]].iloc[-1]["Muscle Group"]
                rw = ss.df.shape[0] + 1
                ss.df.loc[rw] = raw_data
                st.info(f"Added exercise {raw_data['Exercise']}")

with all_exercises_container:
    df = ss.df
    AgGrid(df.reindex(columns=np.roll(df.columns, 1)), height=250, theme="material")

with groups_container:
    if ss.enable_groups:
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
    else:
        groups = {}

#with all_groups_container:
#    if ss.enable_groups:
#        AgGrid(pd.DataFrame(pad_dict_list(groups, "")), theme="material")



with sessions_container:
    def update_sort_key():
        ss.sort_key = datetime.datetime.now().isoformat()

    group_names = [group for group in groups.keys()]
    options = group_names + list(ss.df.Exercise.tolist())
    sessions_with_elements = {
        session: st.multiselect(
            session,
            options=options,
            default=(
                (value for value in ss.default_sessions[session] if value in options)
                if session in ss.default_sessions
                else []
            ),
            key=session+ss.id,
            on_change=update_sort_key()
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

    for session, exercises in sessions_with_exercises.items():
        if len(exercises) < 1:
            st.info(f"You have an empty session, {'remove the empty session or ' if len(sessions)>1 else '' }add some exercises to continue")
            st.stop()

with all_sessions_container:
     table = pd.DataFrame(pad_dict_list(sessions_with_exercises, ""))
     table.index += 1
     AgGrid(table, theme="material")

with session_repeats:
    n_blocks = int(st.number_input("Repeat times", 0, max_value=52,
                                   value=ss.default_blocks, key="blocks"+ss.id))

    f"Congrats. Your program will have {int(n_blocks)} **blocks**. Blocks will look look like this:"

with session_sorting:
    # if st.checkbox("Enable sorting"):
    grid = make_grid(len(sessions) // 3 + 1, 3)
    for i, (session, exercises) in enumerate(sessions_with_exercises.items()):
        with grid[i // 3][i % 3]:
            st.write(session)
            sessions_with_exercises[session] = sort_items(exercises, direction="vertical")# , key=session+ss.sort_key)
    st.info("Drag and drop to **sort** exercises. For **add / remove**, go to step ①")


with add_set_style_container:
    n_col = ss.df_sets.shape[1]  # col count
    rw = -1

    with st.form(key="add progression_form", clear_on_submit=False):
        cols = [st.container()] * n_col  # st.columns(n_col)

        raw_data = {
            "Name": cols[0].text_input("Add Set Style", placeholder="Set Style Name"),
            "Reps": cols[1].text_input("Reps", value="5 / 3 / 1", placeholder="12 / 10 / 8 / 10"),
            "Warmup Weight Adjustment": cols[2].text_input("Weight Adjustment per set", placeholder="-12% / -7.5%")
        }

        # you can insert code for a list comprehension here to change the data
        # values into integer / float, if required

        if st.form_submit_button("Add"):
            if not raw_data["Name"]:
                st.warning("Please name the set style!")
            elif is_set_style(raw_data["Reps"]) and is_weight_adjustment(raw_data["Warmup Weight Adjustment"]):
                df = ss.df_sets
                rw = ss.df_sets.shape[0] + 1
                ss.df_sets.loc[rw] = raw_data
                st.info("Added set style")

with all_set_progressions:
    st.subheader("Set Progressions")
    df = ss.df_progressions
    AgGrid(df, height=250, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS, theme=main_content_table_theme)

with add_progression_container:
    with st.form(key="add set_form", clear_on_submit=False):
        cols = [st.container()] * 2  # st.columns(2)

        raw_data = {
            "Name": cols[0].text_input("Add Progression", placeholder="E.g. accessory progression"),
            "Progression": cols[1].text_input(
                "Progression - weight adjustment for each block",
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
    df = ss.df_sets
    AgGrid(df, height=250, theme=main_content_table_theme, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)

# "## What's your routine bro?"
# with default_set_style_container:
#    default_set_style = st.selectbox("Default set style", options=ss.df_sets.Name, index=6)

with conditional_set_style_container:
    with st.expander("Info"):
        st.info("The prefilled values show how you would configure the Juggernaut program. "
                "See https://liftvault.com/programs/strength/juggernaut-method-base-template-spreadsheet/"
                " for source and details.")
        st.image(PIL.Image.open("assets/juggernaut_programm.jpeg"), use_column_width=True)

    #if st.button(":exclamation: Delete all Rules"):
    #    ss.rule_df = pd.DataFrame()

    sessions_df = pd.DataFrame(
        {
            "Session": sessions,
        }
    )
    blocks_df = pd.DataFrame(
        {
            "Block": list([f"B{i + 1}" for i in range(n_blocks)])
        }
    )
    # session_exercise_df = pd.DataFrame(sessions_with_exercises).add_prefix("Exercise of ")
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

    rule_df_temp_01 = pd.DataFrame(columns=df_2.columns).drop("If Block is", axis=1)

    rule_df_temp_02 = pd.DataFrame({
        "If Block is": blocks_df.Block,
    })

    rule_df_temp_1 = pd.concat([rule_df_temp_02, rule_df_temp_01], axis=1)

with add_rules_container:
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

    select_style_radio = True  # op1
    if len(conditions) > 1:
        select_style_radio = st.checkbox(
            "Same rules for all conditions", True
        )  # st.radio("Select set style", [op1, op2])

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
                        " ".join(np.asarray(condition).flatten()) + "Set Style",
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

with conditional_set_style_container:
    gb = GridOptionsBuilder.from_dataframe(ss.rule_df)
    gb.configure_default_column(
        resizable=True,
        autoHeight=True,
        # wrapText=True,
        width=150,
        cellStyle={
            # 'white-space': 'pre',
            'overflow-wrap': "anywhere",
            "overflow": "visible",
            "text-overflow": "unset",
            "white-space": "normal",
        },
    )
    gb.configure_selection('multiple', use_checkbox=True)
    gb.configure_column(ss.rule_df.columns[0], headerCheckboxSelection=True)

    grid_response = AgGrid(
        ss.rule_df, height=250,
        gridOptions=gb.build(),
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        theme=main_content_table_theme,
    )["selected_rows"]

    if len(grid_response) > 0:
        if st.button("Delete selected"):
            ss.rule_df = ss.rule_df.drop([
                selection["_selectedRowNodeInfo"]["nodeRowIndex"]
                for selection in grid_response
            ]).reset_index(drop=True)
            st.experimental_rerun()

with one_rm_container.form("Enter your 1RMs"):
    st.write("View this 1RM calculator if you don't know your 1RMs exactly.")
    st.image(PIL.Image.open("assets/1RM_calculator.png"))
    one_rms = {}

    plan_structure = {}
    for block_number, block in enumerate(blocks_df["Block"]):
        plan_structure[block] = {}
        for session, exercises in sessions_with_exercises.items():

            for exercise in exercises:
                exercise_row = ss.df.loc[ss.df["Exercise"] == exercise].iloc[-1]

                if exercise not in one_rms:
                    ss.df.loc[(ss.df["Exercise"] == exercise), "1RM"] = one_rms[exercise] = st.number_input(
                        f"1RM for {exercise}",
                        value=exercise_row["1RM"] if "1RM" in exercise_row and str(exercise_row["1RM"]).isnumeric() else 100,
                        key=exercise+"1RM"+ss.id
                    )

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
                        row for index, row in
                        ss.rule_df[ss.rule_df['Block Progression'].notna()].to_dict("index").items()
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
                    set_style_row = ss.df_sets.loc[ss.df_sets["Name"] == current_set_style].iloc[-1]
                else:
                    st.warning(f"{current_set_style} not in set styles")

                reps = [
                    float(rep.strip())
                    for rep in set_style_row['Reps'].split("/")
                ] if set_style_row['Reps'] is not np.nan else []

                weights_percentage = [
                    float(weight.replace("%", "e-2").replace(" ", ""))
                    for weight in set_style_row['Warmup Weight Adjustment'].split("/")
                    if any(filter(str.isdigit, weight))
                ] if set_style_row["Warmup Weight Adjustment"] is not np.nan else []

                def format_set_string(set_format):
                    return "\n".join([ #ss.set_format_delimiter
                        set_format.format(
                            reps=int(rep_count),
                            weight=round(one_rms[exercise] * (progression + weight_percentage)),
                            percentage=(progression + weight_percentage)
                        )
                        for i, (rep_count, weight_percentage)
                        in enumerate(list(itertools.zip_longest(reps, weights_percentage, fillvalue=0)))
                        if set_format != ""
                    ])

                current_set = format_set_string(ss.set_format) + "\n" + format_set_string(ss.set_format_line_2)

                if len(reps) > 0 and exercise not in one_rms and current_set_style != "Empty":
                    one_rms[exercise] = st.number_input(f"1RM for {exercise}", 100)

                plan_structure[block][session + " " + exercise] = current_set

    if st.form_submit_button("Update Plan"):
        pass

with export_plan_container:
    # if True or st.button("Generate Plan"):
    # result_plan = pd.DataFrame.from_dict({
    #       (i,j): plan_structure[i][j]
    #       for i in plan_structure.keys()
    #       for j in plan_structure[i].keys()
    #       for k in plan_structure[i][j].keys()
    #    },
    #    orient="index"
    #  )#.reset_index(level=[0,1]).pivot(index="level_1", columns=["level_0"], values=["Reps", "Weight Percentage"])
    result_plan = pd.DataFrame.from_dict(plan_structure)
    # result_plan =pd.json_normalize(plan_structure, meta=['Blocks','Sessions', "Sets"])#, record_path='Sets')
    result_plan["Exercise"] = result_plan.index
    result_plan = result_plan.reindex(columns=np.roll(result_plan.columns, 1))

    gb = GridOptionsBuilder.from_dataframe(result_plan)

    for column in result_plan.columns[1:]:
        gb.configure_column(
            column,
            resizable=True,
            autoHeight=True,
            maxWidth=75,
            # wrapText=True,
            autoSizeMode=ColumnsAutoSizeMode.FIT_CONTENTS,
            headerTooltip=column,
            cellStyle={
                'overflow-wrap': "anywhere",
                "overflow": "visible",
                "text-overflow": "unset",
                'white-space': "pre", #'normal',
                "font-family": "Courier",
                "font-size": "8pt",
                "line-height": "12px",
                "padding-top": "5px",
                "padding-bottom": "5px"
            },
        )
    gb.configure_column(
        "Exercise",
        pinned=True,
        maxWidth=150,
        cellStyle={
            # 'white-space': 'pre',
            'overflow-wrap': "anywhere",
            "overflow": "visible",
            "text-overflow": "unset",
            "white-space": "normal",
            "font-size": "10pt",
            "line-height": "20px",
            "padding-top": "5px",
            "padding-bottom": "5px"
        },
    )
    AgGrid(
        result_plan,

        gridOptions=gb.build(),
        height=400,

        theme=main_content_table_theme,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS
    )  # columns_auto_size_mode=)
    st.info("Right click the plan to export as .csv / .xlsx")


with settings_container:
    # ss.enable_groups = st.checkbox("Enable exercise grouping")
    ss.set_format = st.text_input(
        "Set format Line 1", ss.set_format,
        help="**Examples**  \n\n"
             "- `{reps}x{weight}kg @{percentage:.1%}`  \n\n"
             "- `{reps}/{weight}lbs`")
    ss.set_format_line_2 = st.text_input(
        "Set format Line 2", ss.set_format_line_2,
        help="**Examples**  \n\n"
             "- `{reps}x{weight}kg @{percentage:.1%}`  \n\n"
             "- `{reps}/{weight}lbs`")

    ss.set_format_delimiter = st.text_input(
        "Set format Delimiter", ss.set_format_delimiter,
        help="**Examples**  \n\n"
             "- `/`  \n\n"
             "- `|`  \n\n"
             "- ` `")
    st.button("Update")

with export_plan_container:
    st.write("**... or save as .gob Template**")
    file_name = st.text_input(
        "Name your template", f"Training",
        help="You can download a savefile as a template so you can continue "
             "working on your plan at a later point.")
    # if st.button("Prepare Savefile"):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:

        ss.df.to_excel(writer, "Exercises", index=False)
        ss.df_sets.to_excel(writer, "Set Styles", index=False)
        ss.df_progressions.to_excel(writer, "Progressions", index=False)
        ss.rule_df.to_excel(writer, "Rules", index=False)
        pd.DataFrame(pad_dict_list(sessions_with_exercises, "")).to_excel(
            writer, "Sessions", index=False
        )
        pd.DataFrame.from_dict({"Number of Blocks": [n_blocks]}).to_excel(
            writer, "Blocks", index=False
        )
        result_plan.to_excel(writer, "Plan", index=False)

        #for sheet_name, sheet in writer.sheets[:-1].items():
        #    sheet.set_column("A:Z", 20)

    st.download_button(
        "Save",#f"Download {datetime.datetime.today().date()}-{file_name}.gob Template",
        buffer,
        file_name=f"{datetime.datetime.today().date()}-{file_name}.gob"
    )

    "## Long-form "
    result_plan.columns.name = "Block"
    result_plan.set_index("Exercise",inplace=True)
    long_result_plan = result_plan.melt(
        value_vars=result_plan.columns,
        value_name="Sets",
        ignore_index=False,
    ).reset_index().set_index(["Block", "Exercise"], drop=True)
    long_result_plan.index = long_result_plan.index.map(" ".join)

    st.write(
        long_result_plan.to_dict(orient="index")
    )


