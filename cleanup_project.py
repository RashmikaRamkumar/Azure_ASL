import config
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from msrest.authentication import ApiKeyCredentials

def cleanup_project():
    """Clean up the existing project by removing unused tags"""
    
    # Initialize client
    credentials = ApiKeyCredentials(in_headers={"Training-key": config.TRAINING_KEY})
    trainer = CustomVisionTrainingClient(config.ENDPOINT, credentials)
    
    # Get the project
    projects = trainer.get_projects()
    project = None
    for p in projects:
        if p.name == "ASL_Gesture_Recognition":
            project = p
            break
    
    if not project:
        print("❌ Project not found!")
        return
    
    print(f"🧹 Cleaning up project: {project.name}")
    print("=" * 50)
    
    # Get all tags
    tags = trainer.get_tags(project.id)
    print(f"Found {len(tags)} tags in project")
    
    # Separate tags by type
    number_tags = []
    uppercase_letter_tags = []
    lowercase_letter_tags = []
    other_tags = []
    
    for tag in tags:
        if tag.name.isdigit():
            number_tags.append(tag)
        elif len(tag.name) == 1 and tag.name.isupper() and tag.name.isalpha():
            uppercase_letter_tags.append(tag)
        elif len(tag.name) == 1 and tag.name.islower() and tag.name.isalpha():
            lowercase_letter_tags.append(tag)
        else:
            other_tags.append(tag)
    
    print(f"\nTag breakdown:")
    print(f"  Numbers (0-9): {len(number_tags)} tags")
    print(f"  Uppercase letters: {len(uppercase_letter_tags)} tags")
    print(f"  Lowercase letters: {len(lowercase_letter_tags)} tags")
    print(f"  Other: {len(other_tags)} tags")
    
    # Check which tags have images
    tags_with_images = []
    empty_tags = []
    
    for tag in tags:
        images = trainer.get_tagged_images(project.id, tag_ids=[tag.id])
        if len(images) > 0:
            tags_with_images.append((tag, len(images)))
            print(f"  ✓ Tag '{tag.name}': {len(images)} images")
        else:
            empty_tags.append(tag)
            print(f"  ⚠️ Tag '{tag.name}': 0 images (empty)")
    
    print(f"\nSummary:")
    print(f"  Tags with images: {len(tags_with_images)}")
    print(f"  Empty tags: {len(empty_tags)}")
    
    # Remove empty tags
    if empty_tags:
        print(f"\n🗑️ Removing {len(empty_tags)} empty tags...")
        for tag in empty_tags:
            try:
                trainer.delete_tag(project.id, tag.id)
                print(f"  ✓ Deleted empty tag: '{tag.name}'")
            except Exception as e:
                print(f"  ❌ Failed to delete tag '{tag.name}': {e}")
    
    print("\n✓ Cleanup completed!")
    
    # Show remaining tags
    remaining_tags = trainer.get_tags(project.id)
    print(f"\nRemaining tags ({len(remaining_tags)}):")
    for tag in remaining_tags:
        images = trainer.get_tagged_images(project.id, tag_ids=[tag.id])
        print(f"  '{tag.name}': {len(images)} images")

def create_fresh_project():
    """Delete the current project and create a fresh one"""
    
    credentials = ApiKeyCredentials(in_headers={"Training-key": config.TRAINING_KEY})
    trainer = CustomVisionTrainingClient(config.ENDPOINT, credentials)
    
    # Find and delete existing project
    projects = trainer.get_projects()
    for project in projects:
        if project.name == "ASL_Gesture_Recognition":
            print(f"🗑️ Deleting existing project: {project.name}")
            try:
                trainer.delete_project(project.id)
                print("✓ Project deleted successfully!")
            except Exception as e:
                print(f"❌ Failed to delete project: {e}")
                return
            break
    
    # Create new project
    print("🆕 Creating fresh project...")
    new_project = trainer.create_project("ASL_Gesture_Recognition")
    print(f"✓ New project created!")
    print(f"  Project ID: {new_project.id}")
    print(f"  Project Name: {new_project.name}")
    
    print(f"\n📝 Update your config.py:")
    print(f'PROJECT_ID = "{new_project.id}"')

def main():
    print("🔧 Azure Custom Vision Project Management")
    print("=" * 50)
    
    choice = input("""
Choose an option:
1. Clean up current project (remove empty tags)
2. Delete current project and create fresh one
3. Exit

Enter choice (1/2/3): """).strip()
    
    if choice == "1":
        cleanup_project()
    elif choice == "2":
        confirm = input("⚠️ This will DELETE all your current data! Are you sure? (yes/no): ")
        if confirm.lower() == "yes":
            create_fresh_project()
        else:
            print("Operation cancelled.")
    elif choice == "3":
        print("Exiting...")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()