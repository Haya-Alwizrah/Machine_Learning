// Skills List --------------------------------------------------

let skills = [];

// Elements ---------------------------------------------

const input = document.getElementById("skillInput");
const tagsContainer = document.getElementById("tagsContainer");
const submitBtn = document.getElementById("submitBtn");
const loading = document.getElementById("loading");
const errorMessage = document.getElementById("errorMessage");
const resultsContainer = document.getElementById("resultsContainer");

// Add Skill ---------------------------------------------

function addSkill() {
    const value = input.value.trim().toLowerCase();
    if (value === "")
        return;
    if (skills.includes(value)) {
        input.value = "";
        return;
    }

    skills.push(value);
    input.value = "";
    renderTags();
}

// Remove Skill ---------------------------------------------

function removeSkill(index){
    skills.splice(index,1);
    renderTags();
}

// Render Tags ---------------------------------------------

function renderTags(){
    tagsContainer.innerHTML = "";
    skills.forEach((skill,index)=>{
        tagsContainer.innerHTML += `
            <div class="tag">
                ${skill}
                <span
                    class="remove-tag"
                    onclick="removeSkill(${index})">
                    &times;
                </span>
            </div>
        `;
    });
}

// Press Enter ---------------------------------------------

input.addEventListener("keypress",function(event){
    if(event.key==="Enter"){
        event.preventDefault();
        addSkill();
    }
});

// Submit Skills ---------------------------------------------

async function submitSkills(){
    errorMessage.innerHTML="";
    resultsContainer.innerHTML="";
    if(skills.length===0){
        errorMessage.innerHTML="Please enter at least one skill.";
        return;
    }

    loading.style.display="block";
    submitBtn.disabled=true;

    try{
        const response = await fetch("/recommend",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({skills:skills})
        });

        const data = await response.json();
        loading.style.display="none";
        submitBtn.disabled=false;

        if(!data.success){
            errorMessage.innerHTML=data.message;
            return;
        }

        displayResults(data.recommendations);
    }

    catch(error){
        loading.style.display="none";
        submitBtn.disabled=false;
        errorMessage.innerHTML="Something went wrong.";
        console.error(error);
    }

}

// Display Results ---------------------------------------------

function displayResults(results){
    resultsContainer.innerHTML="";
    results.forEach(job=>{
        let gapHTML="";
        if(Array.isArray(job.missing_skills)){
            if(job.missing_skills.length===0){
                gapHTML=`
                    <p class="no-gap">
                        No major skill gaps.
                    </p>
                `;
            }

            else{
                job.missing_skills.forEach(skill=>{
                    gapHTML += `
                        <div class="skill-item">
                            ${skill}
                        </div>
                    `;
                });
            }
        }

        else{
            gapHTML=`
                <p class="no-gap">
                    ${job.missing_skills}
                </p>
            `;
        }

        resultsContainer.innerHTML += `
            <div class="job-card">
                <h3>
                    ${job.job_title}
                </h3>

                <!--
                <div class="similarity">
                    Similarity:
                    ${(job.similarity*100).toFixed(1)}%
                </div>
                -->

                <h4>
                    Skill Gap
                </h4>
                <div class="skill-gap">
                    ${gapHTML}
                </div>
            </div>
        `;
    });

    document
        .getElementById("recommendation-section")
        .scrollIntoView({behavior:"smooth"});
}