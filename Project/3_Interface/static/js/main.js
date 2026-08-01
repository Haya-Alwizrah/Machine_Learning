let availableSkills = [];
let userSkills = [];

// Fetch available skills from the server on page load
document.addEventListener('DOMContentLoaded', async function() {
    try {
        const response = await fetch('/api/skills');
        availableSkills = await response.json();
    } catch (error) {
        console.error('Error loading skills:', error);
    }
});

function renderTags() {
    const container = document.getElementById('tagsContainer');
    container.innerHTML = '';
    
    userSkills.forEach(skill => {
        const tag = document.createElement('div');
        tag.className = 'tag';
        
        const skillText = document.createTextNode(skill);
        tag.appendChild(skillText);
        
        const closeBtn = document.createElement('span');
        closeBtn.className = 'close';
        closeBtn.innerText = 'X';
        closeBtn.onclick = function() {
            removeSkill(skill);
        };
        
        tag.appendChild(closeBtn);
        container.appendChild(tag);
    });
}

function removeSkill(skill) {
    userSkills = userSkills.filter(s => s !== skill);
    renderTags();
}

function addSkill(skillValue = null) {
    const input = document.getElementById('skillInput');
    const skill = skillValue || input.value.trim().toLowerCase();
    
    if(skill && !userSkills.includes(skill)) {
        userSkills.push(skill);
        renderTags();
    }
    input.value = ''; 
    input.focus();
    closeSuggestions();
}

// Custom Autocomplete Logic
const skillInput = document.getElementById('skillInput');
const suggestionsBox = document.getElementById('suggestionsBox');

if (skillInput) {
    skillInput.addEventListener('input', function() {
        const val = this.value.trim().toLowerCase();
        closeSuggestions();
        
        if (!val) return false;
        
        const matches = availableSkills.filter(s => s.toLowerCase().includes(val));
        
        if (matches.length > 0) {
            suggestionsBox.style.display = 'block';
            
            matches.slice(0, 15).forEach(match => {
                const div = document.createElement('div');
                div.className = 'suggestion-item';
                div.innerHTML = match;
                div.onclick = function() {
                    addSkill(match);
                };
                suggestionsBox.appendChild(div);
            });
        }
    });

    skillInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            addSkill();
        }
    });
}

function closeSuggestions() {
    if (suggestionsBox) {
        suggestionsBox.innerHTML = '';
        suggestionsBox.style.display = 'none';
    }
}

document.addEventListener('click', function (e) {
    if (e.target.id !== 'skillInput') {
        closeSuggestions();
    }
});

// Submit Logic
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