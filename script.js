// Main JavaScript for AI Engineering Roadmap

// Global state management
let currentView = 'dashboard';
let skillChart = null;
let progressChart = null;
let knowledgeGraphSvg = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeDashboard();
    initializeRoadmap();
    initializeKnowledgeGraph();
    initializeResources();
    initializeProjects();
    initializeProgress();
    
    // Load saved progress
    loadProgress();
});

// Navigation management
function initializeNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    
    navButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            const targetView = e.target.dataset.view;
            switchView(targetView);
        });
    });
}

function switchView(viewName) {
    // Hide all views
    document.querySelectorAll('.view').forEach(view => {
        view.classList.remove('active');
    });
    
    // Show target view
    document.getElementById(viewName).classList.add('active');
    
    // Update navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-view="${viewName}"]`).classList.add('active');
    
    currentView = viewName;
    
    // Refresh view-specific content
    refreshViewContent(viewName);
}

function refreshViewContent(viewName) {
    switch(viewName) {
        case 'dashboard':
            updateDashboardStats();
            break;
        case 'graph':
            if (knowledgeGraphSvg) {
                renderKnowledgeGraph();
            }
            break;
        case 'progress':
            updateProgressCharts();
            break;
    }
}

// Dashboard functionality
function initializeDashboard() {
    updateDashboardStats();
    createSkillChart();
}

function updateDashboardStats() {
    document.getElementById('completedTopics').textContent = progressData.completedTopics.length;
    document.getElementById('currentLevel').textContent = getCurrentLevel();
    document.getElementById('estimatedTime').textContent = calculateRemainingTime();
    
    updateCurrentFocus();
}

function getCurrentLevel() {
    const completed = progressData.completedTopics.length;
    if (completed < 5) return "Beginner";
    if (completed < 15) return "Intermediate";
    if (completed < 30) return "Advanced";
    return "Expert";
}

function calculateRemainingTime() {
    const totalTopics = Object.values(aiLearningData.learningTracks)
        .reduce((sum, track) => sum + track.topics.length, 0);
    const remainingTopics = totalTopics - progressData.completedTopics.length;
    return Math.ceil(remainingTopics * 0.8); // Roughly 0.8 months per topic
}

function updateCurrentFocus() {
    const focusContainer = document.getElementById('currentFocus');
    focusContainer.innerHTML = '';
    
    progressData.currentFocus.forEach(topicId => {
        const topic = findTopicById(topicId);
        if (topic) {
            const progress = Math.random() * 100; // Simulate progress
            const focusItem = createFocusItem(topic.title, progress);
            focusContainer.appendChild(focusItem);
        }
    });
}

function createFocusItem(title, progress) {
    const item = document.createElement('div');
    item.className = 'focus-item';
    item.innerHTML = `
        <span class="focus-topic">${title}</span>
        <div class="progress-bar">
            <div class="progress-fill" style="width: ${progress}%"></div>
        </div>
        <span class="progress-percent">${Math.round(progress)}%</span>
    `;
    return item;
}

function createSkillChart() {
    const ctx = document.getElementById('skillChart').getContext('2d');
    
    skillChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: Object.keys(progressData.skillLevels),
            datasets: [{
                label: 'Current Skills',
                data: Object.values(progressData.skillLevels),
                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                borderColor: 'rgba(102, 126, 234, 1)',
                pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(102, 126, 234, 1)'
            }]
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    angleLines: {
                        display: false
                    },
                    suggestedMin: 0,
                    suggestedMax: 100
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Roadmap functionality
function initializeRoadmap() {
    renderRoadmapTimeline();
    
    document.getElementById('pathFilter').addEventListener('change', renderRoadmapTimeline);
    document.getElementById('toggleParallel').addEventListener('click', toggleParallelTracks);
}

function renderRoadmapTimeline() {
    const container = document.getElementById('roadmapTimeline');
    const filter = document.getElementById('pathFilter').value;
    
    container.innerHTML = '';
    
    const tracks = filter === 'all' ? 
        Object.values(aiLearningData.learningTracks) :
        [aiLearningData.learningTracks[filter]];
    
    tracks.forEach(track => {
        const trackElement = createTrackElement(track);
        container.appendChild(trackElement);
    });
}

function createTrackElement(track) {
    const trackDiv = document.createElement('div');
    trackDiv.className = 'learning-track';
    trackDiv.style.borderLeft = `5px solid ${track.color}`;
    
    const header = document.createElement('div');
    header.className = 'track-header';
    header.innerHTML = `
        <h3>${track.title}</h3>
        <p>${track.description}</p>
        <span class="track-duration">${track.duration}</span>
    `;
    
    const timeline = document.createElement('div');
    timeline.className = 'timeline-track';
    
    track.topics.forEach(topic => {
        const timelineItem = createTimelineItem(topic);
        timeline.appendChild(timelineItem);
    });
    
    trackDiv.appendChild(header);
    trackDiv.appendChild(timeline);
    
    return trackDiv;
}

function createTimelineItem(topic) {
    const item = document.createElement('div');
    item.className = 'timeline-item';
    
    const isCompleted = progressData.completedTopics.includes(topic.id);
    const isInProgress = progressData.currentFocus.includes(topic.id);
    
    if (isCompleted) item.classList.add('completed');
    else if (isInProgress) item.classList.add('in-progress');
    
    item.innerHTML = `
        <div class="timeline-content">
            <div class="timeline-title">${topic.title}</div>
            <div class="timeline-description">${topic.description}</div>
            <div class="timeline-tags">
                ${topic.skills.map(skill => `<span class="tag">${skill}</span>`).join('')}
                <span class="tag importance-${topic.importance}">${topic.importance}</span>
            </div>
        </div>
        <div class="timeline-duration">${topic.duration}</div>
    `;
    
    item.addEventListener('click', () => showTopicDetails(topic));
    
    return item;
}

function showTopicDetails(topic) {
    // Create modal or side panel with detailed topic information
    alert(`Topic: ${topic.title}\n\nDescription: ${topic.description}\n\nDuration: ${topic.duration}\n\nPrerequisites: ${topic.prerequisites.join(', ') || 'None'}`);
}

function toggleParallelTracks() {
    // Implementation for showing parallel learning paths
    console.log('Toggle parallel tracks view');
}

// Knowledge Graph functionality
function initializeKnowledgeGraph() {
    createKnowledgeGraph();
    
    document.getElementById('resetZoom').addEventListener('click', resetGraphZoom);
    document.getElementById('toggleLabels').addEventListener('click', toggleGraphLabels);
    document.getElementById('graphFilter').addEventListener('change', filterGraph);
}

function createKnowledgeGraph() {
    const container = document.getElementById('knowledgeGraph');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    knowledgeGraphSvg = d3.select('#knowledgeGraph')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    renderKnowledgeGraph();
}

function renderKnowledgeGraph() {
    if (!knowledgeGraphSvg) return;
    
    const width = knowledgeGraphSvg.node().clientWidth;
    const height = knowledgeGraphSvg.node().clientHeight;
    
    knowledgeGraphSvg.selectAll('*').remove();
    
    const simulation = d3.forceSimulation(aiLearningData.knowledgeGraph.nodes)
        .force('link', d3.forceLink(aiLearningData.knowledgeGraph.links).id(d => d.id))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2));
    
    const link = knowledgeGraphSvg.append('g')
        .selectAll('line')
        .data(aiLearningData.knowledgeGraph.links)
        .enter().append('line')
        .attr('stroke', '#999')
        .attr('stroke-opacity', 0.6)
        .attr('stroke-width', 2);
    
    const node = knowledgeGraphSvg.append('g')
        .selectAll('g')
        .data(aiLearningData.knowledgeGraph.nodes)
        .enter().append('g')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    const circles = node.append('circle')
        .attr('r', d => 20 + d.level * 5)
        .attr('fill', d => getNodeColor(d.group))
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);
    
    const labels = node.append('text')
        .text(d => d.id.replace('-', ' '))
        .attr('text-anchor', 'middle')
        .attr('dy', 4)
        .style('font-size', '10px')
        .style('fill', '#333');
    
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node
            .attr('transform', d => `translate(${d.x},${d.y})`);
    });
    
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

function getNodeColor(group) {
    const colors = {
        'math': '#48bb78',
        'cs': '#667eea',
        'programming': '#ed8936',
        'ml': '#805ad5',
        'dl': '#f56565',
        'advanced': '#38b2ac',
        'agi': '#d69e2e'
    };
    return colors[group] || '#718096';
}

function resetGraphZoom() {
    // Implementation for resetting zoom
    renderKnowledgeGraph();
}

function toggleGraphLabels() {
    // Implementation for toggling labels visibility
    const labels = knowledgeGraphSvg.selectAll('text');
    const currentOpacity = labels.style('opacity');
    labels.style('opacity', currentOpacity === '0' ? '1' : '0');
}

function filterGraph() {
    // Implementation for filtering graph by category
    const filter = document.getElementById('graphFilter').value;
    renderKnowledgeGraph(); // Re-render with filter applied
}

// Resources functionality
function initializeResources() {
    renderResources();
    
    document.getElementById('resourceSearch').addEventListener('input', filterResources);
    document.getElementById('resourceFilter').addEventListener('change', filterResources);
}

function renderResources() {
    const container = document.getElementById('resourcesGrid');
    container.innerHTML = '';
    
    aiLearningData.resources.forEach(resource => {
        const resourceElement = createResourceElement(resource);
        container.appendChild(resourceElement);
    });
}

function createResourceElement(resource) {
    const item = document.createElement('div');
    item.className = 'resource-item';
    item.innerHTML = `
        <div class="resource-header">
            <div>
                <div class="resource-title">${resource.title}</div>
                <div style="color: #718096; font-size: 0.9rem;">${resource.author}</div>
            </div>
            <span class="resource-type">${resource.type}</span>
        </div>
        <div class="resource-description">${resource.description}</div>
        <div class="resource-meta">
            <span>Difficulty: ${resource.difficulty}</span>
            <span>Rating: ⭐ ${resource.rating}</span>
        </div>
    `;
    
    item.addEventListener('click', () => {
        window.open(resource.url, '_blank');
    });
    
    return item;
}

function filterResources() {
    const searchTerm = document.getElementById('resourceSearch').value.toLowerCase();
    const typeFilter = document.getElementById('resourceFilter').value;
    
    const filteredResources = aiLearningData.resources.filter(resource => {
        const matchesSearch = resource.title.toLowerCase().includes(searchTerm) ||
                            resource.description.toLowerCase().includes(searchTerm);
        const matchesType = typeFilter === 'all' || resource.type === typeFilter;
        
        return matchesSearch && matchesType;
    });
    
    const container = document.getElementById('resourcesGrid');
    container.innerHTML = '';
    
    filteredResources.forEach(resource => {
        const resourceElement = createResourceElement(resource);
        container.appendChild(resourceElement);
    });
}

// Projects functionality
function initializeProjects() {
    renderProjects();
    
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            filterProjects(e.target.dataset.difficulty);
        });
    });
}

function renderProjects(filter = 'all') {
    const container = document.getElementById('projectsGrid');
    container.innerHTML = '';
    
    const filteredProjects = filter === 'all' ? 
        aiLearningData.projects :
        aiLearningData.projects.filter(project => project.difficulty === filter);
    
    filteredProjects.forEach(project => {
        const projectElement = createProjectElement(project);
        container.appendChild(projectElement);
    });
}

function createProjectElement(project) {
    const card = document.createElement('div');
    card.className = 'project-card';
    card.innerHTML = `
        <div class="project-header">
            <div class="project-title">${project.title}</div>
            <span class="difficulty-badge difficulty-${project.difficulty}">${project.difficulty}</span>
        </div>
        <div class="project-description">${project.description}</div>
        <div class="project-skills">
            ${project.skills.map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
        </div>
        <div class="project-footer">
            <span>Duration: ${project.duration}</span>
            <span>${project.topics.length} topics covered</span>
        </div>
    `;
    
    card.addEventListener('click', () => showProjectDetails(project));
    
    return card;
}

function showProjectDetails(project) {
    // Create detailed project view
    const details = `
Project: ${project.title}

Description: ${project.description}

Skills Required: ${project.skills.join(', ')}

Duration: ${project.duration}

Learning Objectives:
${project.learningObjectives ? project.learningObjectives.map(obj => `• ${obj}`).join('\n') : 'Not specified'}

Topics Covered: ${project.topics.join(', ')}
    `;
    
    alert(details);
}

function filterProjects(difficulty) {
    renderProjects(difficulty);
}

// Progress functionality
function initializeProgress() {
    renderMilestones();
    renderTopicProgress();
    createProgressChart();
}

function renderMilestones() {
    const container = document.getElementById('milestonesTimeline');
    container.innerHTML = '';
    
    aiLearningData.milestones.forEach(milestone => {
        const milestoneElement = createMilestoneElement(milestone);
        container.appendChild(milestoneElement);
    });
}

function createMilestoneElement(milestone) {
    const element = document.createElement('div');
    element.className = `milestone ${milestone.status}`;
    
    let icon = '⏳';
    if (milestone.status === 'completed') icon = '✓';
    else if (milestone.status === 'current') icon = '🎯';
    
    element.innerHTML = `
        <div class="milestone-icon">${icon}</div>
        <div class="milestone-content">
            <div class="milestone-title">${milestone.title}</div>
            <div class="milestone-date">${milestone.estimatedDate}</div>
        </div>
    `;
    
    return element;
}

function renderTopicProgress() {
    const container = document.getElementById('topicsProgress');
    container.innerHTML = '';
    
    // Get all topics from all tracks
    const allTopics = [];
    Object.values(aiLearningData.learningTracks).forEach(track => {
        track.topics.forEach(topic => {
            allTopics.push({...topic, track: track.title});
        });
    });
    
    allTopics.forEach(topic => {
        const topicElement = createTopicProgressElement(topic);
        container.appendChild(topicElement);
    });
}

function createTopicProgressElement(topic) {
    const element = document.createElement('div');
    element.className = 'topic-progress';
    
    const isCompleted = progressData.completedTopics.includes(topic.id);
    const isInProgress = progressData.currentFocus.includes(topic.id);
    
    let progress = 0;
    if (isCompleted) progress = 100;
    else if (isInProgress) progress = Math.random() * 80 + 10; // Simulate progress
    
    element.innerHTML = `
        <div class="topic-header">
            <span class="topic-name">${topic.title}</span>
            <span class="topic-percent">${Math.round(progress)}%</span>
        </div>
        <div class="topic-progress-bar">
            <div class="topic-progress-fill" style="width: ${progress}%"></div>
        </div>
        <div class="topic-subtopics">${topic.track} • ${topic.duration}</div>
    `;
    
    return element;
}

function createProgressChart() {
    const ctx = document.getElementById('progressChart').getContext('2d');
    
    const trackNames = Object.keys(aiLearningData.learningTracks);
    const trackProgress = trackNames.map(trackName => {
        const track = aiLearningData.learningTracks[trackName];
        const completedInTrack = track.topics.filter(topic => 
            progressData.completedTopics.includes(topic.id)
        ).length;
        return (completedInTrack / track.topics.length) * 100;
    });
    
    progressChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: trackNames.map(name => name.replace(/([A-Z])/g, ' $1').trim()),
            datasets: [{
                label: 'Progress %',
                data: trackProgress,
                backgroundColor: [
                    '#48bb78',
                    '#667eea',
                    '#ed8936',
                    '#805ad5',
                    '#f56565',
                    '#38b2ac'
                ],
                borderColor: [
                    '#38a169',
                    '#5a67d8',
                    '#dd6b20',
                    '#6b46c1',
                    '#e53e3e',
                    '#319795'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

function updateProgressCharts() {
    if (progressChart) {
        // Update progress chart with current data
        const trackNames = Object.keys(aiLearningData.learningTracks);
        const trackProgress = trackNames.map(trackName => {
            const track = aiLearningData.learningTracks[trackName];
            const completedInTrack = track.topics.filter(topic => 
                progressData.completedTopics.includes(topic.id)
            ).length;
            return (completedInTrack / track.topics.length) * 100;
        });
        
        progressChart.data.datasets[0].data = trackProgress;
        progressChart.update();
    }
}

// Utility functions
function findTopicById(topicId) {
    for (const track of Object.values(aiLearningData.learningTracks)) {
        const topic = track.topics.find(t => t.id === topicId);
        if (topic) return topic;
    }
    return null;
}

function markTopicCompleted(topicId) {
    if (!progressData.completedTopics.includes(topicId)) {
        progressData.completedTopics.push(topicId);
        saveProgress();
        updateDashboardStats();
        updateProgressCharts();
    }
}

function addToCurrentFocus(topicId) {
    if (!progressData.currentFocus.includes(topicId)) {
        progressData.currentFocus.push(topicId);
        saveProgress();
        updateCurrentFocus();
    }
}

// Progress persistence
function saveProgress() {
    localStorage.setItem('aiLearningProgress', JSON.stringify(progressData));
}

function loadProgress() {
    const saved = localStorage.getItem('aiLearningProgress');
    if (saved) {
        const savedProgress = JSON.parse(saved);
        Object.assign(progressData, savedProgress);
    }
}

// Export functions for external use
window.aiRoadmap = {
    markTopicCompleted,
    addToCurrentFocus,
    switchView,
    progressData,
    aiLearningData
};