let userSkills = [];

function addSkill() {
    const input = document.getElementById('skillInput');
    const skill = input.value.trim().toLowerCase();
    
    if(skill && !userSkills.includes(skill)) {
        userSkills.push(skill);
        renderTags();
    }
    input.value = ''; 
    input.focus();
}

function removeSkill(skill) {
    userSkills = userSkills.filter(s => s !== skill);
    renderTags();
}

function renderTags() {
    const container = document.getElementById('tagsContainer');
    container.innerHTML = '';
    userSkills.forEach(skill => {
        const tag = document.createElement('div');
        tag.className = 'tag';
        tag.innerHTML = `${skill} <span class="close" onclick="removeSkill('${skill}')">X</span>`;
        container.appendChild(tag);
    });
}

const skillInput = document.getElementById('skillInput');
if (skillInput) {
    skillInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            addSkill();
        }
    });
}

async function submitSkills() {
    if(userSkills.length === 0) {
        alert("Please enter at least one skill.");
        return;
    }

    const btn = document.getElementById('submitBtn');
    btn.innerText = "Analyzing...";
    btn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ skills: userSkills })
        });
        
        const data = await response.json();
        
        const resultsSection = document.getElementById('resultsSection');
        const jobsList = document.getElementById('jobsList');
        const missingSkillsContainer = document.getElementById('missingSkillsContainer');
        
        jobsList.innerHTML = data.jobs.map(job => `<li><strong>${job}</strong></li>`).join('');
        missingSkillsContainer.innerHTML = data.missing_skills.map(s => `<span class="skill-badge">${s}</span>`).join('');
        
        resultsSection.style.display = 'block';
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        alert("An error occurred while generating recommendations.");
    } finally {
        btn.innerText = "Get Recommendations";
        btn.disabled = false;
    }
}